# NixOS module: one rank of the two-Spark server as a systemd service, dsv41-tensorfold.service, on each Spark.
#   role = "head":   rank 0, tensorfold-native serve (the OpenAI API on `port`)
#   role = "worker": rank 1, tf-dsv41-lanes following rank 0
# The service is not started at boot (no wantedBy) and a `nixos-rebuild switch` does not restart it while it runs:
#   worker:  systemctl start dsv41-tensorfold     (first: it waits for rank 0's rendezvous)
#   head:    systemctl start dsv41-tensorfold     (returns once the API answers, `waitReady`)
#   both:    systemctl stop dsv41-tensorfold      (either rank ending leaves the other waiting: stop both)
# The files the ranks read are bin/prepare's (run it once from the head as `user`; the paths below default to its layout).
# What docker gave the ranks is the unit's: LimitMEMLOCK=infinity (RDMA), LimitSTACK=64M. The memory guard
# (scripts/memguard.sh) runs beside each rank as dsv41-tensorfold-memguard.service and kills the rank before a Spark freezes.
# The settings that both ranks must share (context, parallel, drafts, rounds, graphs ...) are best set once, in a file
# both hosts import.
{ self, pins }:
{ config, lib, pkgs, ... }:
let
  cfg = config.services.dsv41-tensorfold;
  inherit (lib) mkOption mkEnableOption mkIf types optional optionals optionalString concatStringsSep escapeShellArg escapeShellArgs;
  rev8 = builtins.substring 0 8;
  home = (config.users.users.${cfg.user} or { home = "/home/${cfg.user}"; }).home;
  isHead = cfg.role == "head";
  b = x: if x then "1" else "0";

  engramId = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
  engramRev = "1bae0e1088993b9b82de08f59de5be5a34e42170";

  # TF_DS_* both ranks read (BertholomusAI's two-node launch), and yours
  engineEnv = {
    TF_DS_REPLAY_FLOOR = b cfg.replayFloor;
    TF_DS_STREAMS = toString cfg.parallel;
    HF_HUB_OFFLINE = "1";
  } // lib.optionalAttrs (cfg.roundRows != null) { TF_DS_ROUND_ROWS = toString cfg.roundRows; }
    // lib.optionalAttrs (cfg.driverLibDir != null) { TF_DRIVER_LIB_DIR = cfg.driverLibDir; }
    // cfg.environment;

  serveArgs = [
    "serve" cfg.head.servedDir "--backend" "cuda" "--host" cfg.head.host "--port" (toString cfg.port) "--name" cfg.servedName
    "--context" (toString cfg.context) "--parallel" (toString cfg.parallel) "--max-tokens" (toString cfg.maxTokens)
    "--temperature" (toString cfg.temperature) "--top-p" (toString cfg.topP) "--top-k" (toString cfg.topK)
    (if cfg.thinking then "--thinking" else "--no-thinking")
  ] ++ optionals (cfg.reasoningEffort != null) [ "--reasoning-effort" cfg.reasoningEffort ]
    ++ optionals (cfg.thinkingBudget != null) [ "--thinking-budget" (toString cfg.thinkingBudget) ]
    ++ optional (!cfg.drafts) "--no-drafts"
    ++ optionals (cfg.head.apiKeyFile != null) [ "--api-key-file" cfg.head.apiKeyFile ]
    ++ cfg.head.extraArgs;

  # rank 1 takes the round's settings as flags; its pool is rank 0's context and its streams rank 0's
  followArgs = [
    "--pool" (toString cfg.context) "--streams" (toString cfg.parallel) "--drafts" (b cfg.drafts) "--graphs" (b cfg.graphs)
    "--side" (b cfg.side) "--prefetch" (b cfg.prefetch) "--aio" (b cfg.engramAio)
  ] ++ optionals (cfg.roundRows != null) [ "--round-rows" (toString cfg.roundRows) ]
    ++ cfg.worker.extraArgs;

  # The link: this node's netdev, RoCE devices and RoCE v2 GID index toward the peer (scripts/nodes.sh's link_info and
  # rails, as bin/start finds them), unless all three are given
  linkScript = ''
    log()  { echo "dsv41: $*"; }
    warn() { echo "dsv41: WARN: $*" >&2; }
    die()  { echo "dsv41: ERROR: $*" >&2; exit 1; }
    source ${../scripts/nodes.sh}
    PEER=${escapeShellArg cfg.peerAddress}
    read -r ADDR DEV HCA GID <<<"$(link_info "$PEER")" || true
    [[ -n "''${ADDR:-}" ]] || die "no route to the peer $PEER (peerAddress: the other Spark's address on the CX7 link)"
    DEV=${if cfg.network.socketIfname != null then escapeShellArg cfg.network.socketIfname else ''"$DEV"''}
    GID=${if cfg.network.gidIndex != null then toString cfg.network.gidIndex else ''"$GID"''}
    ${if cfg.network.hcas != null then "HCAS=${escapeShellArg cfg.network.hcas}"
      else if cfg.network.rails == 1 then ''HCAS="$HCA"''
      else ''HCAS=$(rails "$DEV" "$GID")''}
    [[ "$HCAS" != - && -n "$HCAS" && "$GID" != - ]] ||
      die "no RoCE device or RoCE v2 GID on $DEV toward $PEER: is it the CX7 port between the Sparks? (network.hcas / network.gidIndex)"
    log "link: $ADDR ($DEV) -> $PEER, RoCE $HCAS, GID index $GID"
    NCCL=(NCCL_SOCKET_IFNAME="$DEV" NCCL_IB_HCA="$HCAS" NCCL_IB_GID_INDEX="$GID")
  '';

  # what rank 0 alone reads (bin/start's ENV0, the link's part aside)
  rank0Env = {
    TF_DS_RANK_CACHE = cfg.head.rankCacheDir;
    TF_DS_KIT = cfg.head.kitDir;
    TENSORFOLD_CUDA_KERNELS = "${cfg.head.kitDir}/aot";
    TF_DS_ENGRAM = cfg.head.engramDir;
    TF_DS_TOKEN_MAP = cfg.head.tokenMap;
    TF_DS_GRAPHS = b cfg.graphs;
    TF_DS_HC_SIDE = b cfg.side;
    TF_DS_L2_PREFETCH = b cfg.prefetch;
    TF_DS_ENGRAM_AIO = b cfg.engramAio;
  } // lib.optionalAttrs (cfg.head.prefillInterleaveMs != 0) { TF_DS_PREFILL_INTERLEAVE_MS = toString cfg.head.prefillInterleaveMs; };

  envWords = attrs: escapeShellArgs (lib.mapAttrsToList (n: v: "${n}=${v}") attrs);

  headStart = pkgs.writeShellScript "dsv41-rank0" ''
    set -euo pipefail
    ${linkScript}
    for f in ${escapeShellArgs [ "${cfg.head.servedDir}/config.json" cfg.head.kitDir cfg.head.tokenMap "${cfg.head.engramDir}/config.json" ]}; do
      [[ -e "$f" ]] || die "$f is missing: run bin/prepare from this Spark (as ${cfg.user})"
    done
    compgen -G ${escapeShellArg "${cfg.head.rankCacheDir}/rank0of2-*.bin"} >/dev/null || die "no rank 0 weight file in ${cfg.head.rankCacheDir}: run bin/prepare"
    exec env ${envWords engineEnv} "''${NCCL[@]}" \
      TF_TP_WORLD=2 TF_TP_MASTER="$ADDR" TF_TP_PORT=${toString cfg.masterPort} \
      ${envWords rank0Env} \
      TF_RDMA_GID_INDEX="$GID" ${optionalString cfg.rdma ''TF_RDMA_DEVICES="$HCAS"''} \
      ${cfg.package}/bin/tensorfold-native ${escapeShellArgs serveArgs}
  '';

  workerStart = pkgs.writeShellScript "dsv41-rank1" ''
    set -euo pipefail
    ${linkScript}
    for f in ${escapeShellArgs [ "${cfg.worker.modelDir}/config.json" cfg.worker.kitDir cfg.worker.tokenMap ]}; do
      [[ -e "$f" ]] || die "$f is missing: run bin/prepare on the head (it copies the worker's files)"
    done
    compgen -G ${escapeShellArg "${cfg.worker.rankCacheDir}/rank1of2-*.bin"} >/dev/null || die "no rank 1 weight file in ${cfg.worker.rankCacheDir}: run bin/prepare on the head"
    exec env ${envWords engineEnv} "''${NCCL[@]}" TF_RDMA_GID_INDEX="$GID" \
      ${cfg.package}/bin/tf-dsv41-lanes ${escapeShellArgs [ cfg.worker.modelDir cfg.worker.rankCacheDir ]} 1 2 "$PEER" ${toString cfg.masterPort} \
      ${escapeShellArgs [ cfg.worker.kitDir "--engram" cfg.worker.engramDir "--token-map" cfg.worker.tokenMap ]} \
      ${escapeShellArgs followArgs} ${optionalString cfg.rdma ''--rdma "$HCAS"''}
  '';

  # a start needs the memory: the weights (98.9 GiB) and the engine's reserve (12.2 GiB); re-read for a while, because page
  # cache a large copy just left is counted as available only as it is written back
  memoryCheck = pkgs.writeShellScript "dsv41-memory" ''
    for i in $(seq 0 2 ${toString cfg.minAvailWaitSeconds}); do
      a=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
      (( a >= ${toString cfg.minAvailGiB} )) && exit 0
      sleep 2
    done
    echo "dsv41: only $a GiB memory available, ${toString cfg.minAvailGiB} GiB needed (minAvailGiB): a rank that takes all of a Spark's memory freezes it" >&2
    exit 1
  '';

  headReady = pkgs.writeShellScript "dsv41-ready" ''
    set -u
    auth=()
    ${optionalString (cfg.head.apiKeyFile != null) ''
      printf 'Authorization: Bearer %s\n' "$(head -n 1 ${escapeShellArg cfg.head.apiKeyFile} | sed 's/^[^:]*: *//')" > "$RUNTIME_DIRECTORY/auth"
      auth=(-H "@$RUNTIME_DIRECTORY/auth")
    ''}
    host=${escapeShellArg cfg.head.host}; [[ "$host" == 0.0.0.0 || "$host" == "::" ]] && host=127.0.0.1
    until curl -sf --max-time 5 "''${auth[@]}" "http://$host:${toString cfg.port}/v1/models" >/dev/null; do sleep 5; done
    echo "dsv41: ${cfg.servedName} answers on port ${toString cfg.port}"
  '';

  memguardStart = pkgs.writeShellScript "dsv41-memguard" ''
    set -u
    pid=0
    for i in $(seq 1 600); do
      pid=$(systemctl show -p MainPID --value dsv41-tensorfold.service)
      [[ "$pid" != 0 ]] && break
      sleep 1
    done
    [[ "$pid" != 0 ]] || { echo "dsv41-memguard: the rank did not start" >&2; exit 1; }
    echo "$pid" > "$RUNTIME_DIRECTORY/rank.pid"
    exec ${pkgs.bash}/bin/bash ${../scripts/memguard.sh} run "pidfile:$RUNTIME_DIRECTORY/rank.pid" \
      ${toString cfg.memguard.floorGiB} "$LOGS_DIRECTORY/memguard.log" "$RUNTIME_DIRECTORY/memguard.pid" \
      ${toString cfg.memguard.holdSeconds} ${toString cfg.memguard.hardFloorGiB}
  '';
in
{
  options.services.dsv41-tensorfold = {
    enable = mkEnableOption "one rank of DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw on two DGX Sparks (TensorFold), as dsv41-tensorfold.service";

    package = mkOption {
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.tensorfold;
      defaultText = lib.literalExpression "dsv41-flake.packages.\${system}.tensorfold";
      description = "The engine (the flake's TensorFold build). Both Sparks must run the same one.";
    };

    role = mkOption {
      type = types.enum [ "head" "worker" ];
      description = ''"head": rank 0 and the API. "worker": rank 1.'';
    };

    user = mkOption {
      type = types.str;
      example = "curt";
      description = "The user the rank runs as: the one who ran bin/prepare, whose files it reads.";
    };
    group = mkOption { type = types.str; default = "users"; description = "The rank's group."; };

    peerAddress = mkOption {
      type = types.str;
      example = "192.168.100.11";
      description = ''
        The other Spark's address on the CX7 link. On the worker this is rank 0's rendezvous address
        (TF_TP_MASTER); on the head it picks the link (its own address on it is the rendezvous).
      '';
    };
    masterPort = mkOption { type = types.port; default = 29620; description = "The ranks' rendezvous port (MASTER_PORT)."; };

    network = {
      rails = mkOption {
        type = types.enum [ 1 2 ];
        default = 2;
        description = ''
          2: every RoCE device on the link (the cabled port's PCIe twin too: a prompt chunk's all-gather is ~1.8x faster),
          as bin/start finds them; 1: the link's own device only (NCCL_RAILS=1). Use 1 when the twins are not addressed
          on one subnet per rail with the same MTU on both Sparks (bin/start probes that; this unit does not).
        '';
      };
      socketIfname = mkOption { type = types.nullOr types.str; default = null; description = "NCCL_SOCKET_IFNAME (null: the netdev of the route to peerAddress)."; };
      hcas = mkOption { type = types.nullOr types.str; default = null; example = "rocep1s0f1,roceP2p1s0f1"; description = "NCCL_IB_HCA and the RDMA devices (null: found as `rails` says)."; };
      gidIndex = mkOption { type = types.nullOr types.int; default = null; description = "NCCL_IB_GID_INDEX (null: the RoCE v2 GID holding this node's link address)."; };
      trustedInterfaces = mkOption {
        type = types.listOf types.str;
        default = [ ];
        example = [ "enp1s0f1np1" "enP2p1s0f1np1" ];
        description = "The CX7 link's netdevs, added to networking.firewall.trustedInterfaces (the rendezvous and NCCL's own ports).";
      };
    };

    # ---------------------------------------------------------------- serving (scripts/config.sh's defaults)
    port = mkOption { type = types.port; default = 8888; description = "The API's port (head)."; };
    openFirewall = mkOption { type = types.bool; default = false; description = "Open `port` in the firewall (head)."; };
    servedName = mkOption { type = types.str; default = "DeepSeek-V4.1-Flash-1017"; description = "The model name the API serves."; };
    context = mkOption { type = types.ints.between 1 1048576; default = 1048576; description = "Positions a request may use (both ranks)."; };
    parallel = mkOption { type = types.ints.between 1 16; default = 4; description = "Requests at once (both ranks)."; };
    roundRows = mkOption { type = types.nullOr (types.ints.between 16 48); default = null; description = "A decode round's rows at most (both ranks; null: 16)."; };
    maxTokens = mkOption { type = types.ints.positive; default = 32768; description = "A reply's budget when a request sets none."; };
    thinking = mkOption { type = types.bool; default = true; description = "Think first unless a request turns it off."; };
    reasoningEffort = mkOption { type = types.nullOr (types.enum [ "low" "medium" "high" "xhigh" ]); default = null; description = "Reasoning effort when a request names none."; };
    thinkingBudget = mkOption { type = types.nullOr types.ints.positive; default = null; description = "Reasoning tokens before the engine closes the think block (null: no cap)."; };
    temperature = mkOption { type = types.str; default = "1.0"; description = "Default temperature."; };
    topP = mkOption { type = types.str; default = "0.95"; description = "Default top_p."; };
    topK = mkOption { type = types.ints.positive; default = 20; description = "Default top_k."; };
    drafts = mkOption { type = types.bool; default = true; description = "DSpark draft blocks (both ranks)."; };
    rdma = mkOption { type = types.bool; default = true; description = "Decode-size exchanges over RoCE instead of NCCL (both ranks)."; };
    graphs = mkOption { type = types.bool; default = true; description = "CUDA graphs (both ranks)."; };
    side = mkOption { type = types.bool; default = true; description = "The mixes' side stream (both ranks)."; };
    prefetch = mkOption { type = types.bool; default = true; description = "Paced L2 prefetch (both ranks)."; };
    engramAio = mkOption { type = types.bool; default = true; description = "Engram reads by Linux AIO (both ranks)."; };
    replayFloor = mkOption { type = types.bool; default = true; description = "TF_DS_REPLAY_FLOOR (both ranks)."; };
    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      example = { TF_DS_IDX_SPLIT = "1"; NCCL_DEBUG = "WARN"; };
      description = "More environment for the rank (TF_DS_*, TENSORFOLD_*, NCCL_*: zig/patches/README.md). Give both ranks the same.";
    };
    driverLibDir = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "The folder of the NVIDIA driver's libcuda.so.1 (null: /run/opengl-driver/lib, NixOS's).";
    };

    # ---------------------------------------------------------------- the files (bin/prepare's layout)
    modelRevision = mkOption {
      type = types.str;
      default = pins.model;
      description = "The checkpoint's revision bin/prepare prepared (the flake's pin): it names the data folders.";
    };
    head = {
      hfCache = mkOption { type = types.str; default = "${home}/.cache/huggingface"; defaultText = "~user/.cache/huggingface"; description = "HF_CACHE on the head."; };
      dataDir = mkOption {
        type = types.str;
        default = "${cfg.head.hfCache}/tensorfold-dsv41/${rev8 cfg.modelRevision}";
        defaultText = "<hfCache>/tensorfold-dsv41/<revision>";
        description = "bin/prepare's DATA_DIR (append -ablit-<rev> for ABLIT=1).";
      };
      servedDir = mkOption { type = types.str; default = "${cfg.head.dataDir}/model"; defaultText = "<dataDir>/model"; description = "The folder rank 0 serves."; };
      rankCacheDir = mkOption { type = types.str; default = "${cfg.head.dataDir}/rank-cache"; defaultText = "<dataDir>/rank-cache"; description = "Rank 0's weight file's folder."; };
      tokenMap = mkOption { type = types.str; default = "${cfg.head.dataDir}/dsv41_token_map.json"; defaultText = "<dataDir>/dsv41_token_map.json"; description = "Engram's token map."; };
      kitDir = mkOption { type = types.str; default = "${cfg.head.hfCache}/tensorfold-dsv41/kit-${rev8 pins.kit}/kit"; defaultText = "<hfCache>/tensorfold-dsv41/kit-<rev>/kit"; description = "The kernel kit."; };
      engramDir = mkOption {
        type = types.str;
        default = "${cfg.head.hfCache}/hub/models--${lib.replaceStrings [ "/" ] [ "--" ] engramId}/snapshots/${engramRev}";
        defaultText = "<hfCache>/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<rev>";
        description = "Engram's tables (HEAD_ENGRAM_DIR: a copy on a local disk is faster than over NFS).";
      };
      host = mkOption { type = types.str; default = "0.0.0.0"; description = "The API's address."; };
      apiKeyFile = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Require a key (tensorfold-native --api-key-file; one key a line, reread when it changes). Readable by `user`.";
      };
      prefillInterleaveMs = mkOption { type = types.ints.between 0 5000; default = 250; description = "TF_DS_PREFILL_INTERLEAVE_MS (rank 0)."; };
      waitReady = mkOption {
        type = types.bool;
        default = true;
        description = "`systemctl start` returns once the API answers (a few minutes), not when the process starts.";
      };
      readyTimeout = mkOption { type = types.ints.positive; default = 2400; description = "Seconds the load may take (WAIT_TIMEOUT)."; };
      extraArgs = mkOption { type = types.listOf types.str; default = [ ]; description = "More `tensorfold-native serve` arguments (they win)."; };
    };
    worker = {
      dataDir = mkOption {
        type = types.str;
        default = "${home}/.cache/dsv41-tensorfold/data-${rev8 cfg.modelRevision}";
        defaultText = "~user/.cache/dsv41-tensorfold/data-<revision>";
        description = "bin/prepare's WORKER_DATA_DIR (the checkpoint's config, the kit, the token map).";
      };
      modelDir = mkOption { type = types.str; default = "${cfg.worker.dataDir}/model"; defaultText = "<dataDir>/model"; description = "The checkpoint's files (rank 1 reads config.json)."; };
      kitDir = mkOption { type = types.str; default = "${cfg.worker.dataDir}/kit"; defaultText = "<dataDir>/kit"; description = "The kernel kit."; };
      tokenMap = mkOption { type = types.str; default = "${cfg.worker.dataDir}/dsv41_token_map.json"; defaultText = "<dataDir>/dsv41_token_map.json"; description = "Engram's token map."; };
      engramDir = mkOption { type = types.str; default = "${home}/.cache/dsv41-tensorfold/engram-${rev8 engramRev}"; defaultText = "~user/.cache/dsv41-tensorfold/engram-<rev>"; description = "WORKER_ENGRAM_DIR."; };
      rankCacheDir = mkOption { type = types.str; default = "${home}/.cache/dsv41-tensorfold/rank-cache-${rev8 cfg.modelRevision}"; defaultText = "~user/.cache/dsv41-tensorfold/rank-cache-<revision>"; description = "WORKER_CACHE_DIR (rank 1's weight file)."; };
      extraArgs = mkOption { type = types.listOf types.str; default = [ ]; description = "More tf-dsv41-lanes arguments (e.g. [ \"--profile\" \"1\" ])."; };
    };

    # ---------------------------------------------------------------- memory
    minAvailGiB = mkOption { type = types.ints.unsigned; default = 112; description = "Refuse to start under this much MemAvailable (0: off)."; };
    minAvailWaitSeconds = mkOption { type = types.ints.unsigned; default = 30; description = "Re-read MemAvailable for this long before refusing."; };
    memguard = {
      enable = mkOption { type = types.bool; default = true; description = "dsv41-tensorfold-memguard.service beside the rank (scripts/memguard.sh)."; };
      floorGiB = mkOption { type = types.str; default = "2.5"; description = "Kill the rank when MemAvailable stays under this for holdSeconds."; };
      holdSeconds = mkOption { type = types.str; default = "4"; description = "Seconds under the floor before a kill."; };
      hardFloorGiB = mkOption { type = types.str; default = "1"; description = "Kill at once under this."; };
    };

    rdmaSupport = mkOption {
      type = types.bool;
      default = true;
      description = "rdma-core's udev rules (/dev/infiniband/uverbs* for users) and the RoCE kernel modules (ib_uverbs, rdma_ucm).";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      { assertion = cfg.peerAddress != ""; message = "services.dsv41-tensorfold.peerAddress: the other Spark's address on the CX7 link"; }
    ];
    warnings = optional (!pkgs.stdenv.hostPlatform.isAarch64) "services.dsv41-tensorfold: a DGX Spark is aarch64-linux";

    services.udev.packages = optional cfg.rdmaSupport pkgs.rdma-core;
    boot.kernelModules = optionals cfg.rdmaSupport [ "ib_uverbs" "rdma_ucm" ];
    networking.firewall.allowedTCPPorts = optional (isHead && cfg.openFirewall) cfg.port;
    networking.firewall.trustedInterfaces = cfg.network.trustedInterfaces;

    systemd.services.dsv41-tensorfold = {
      description = "DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw, ${if isHead then "rank 0 (API on port ${toString cfg.port})" else "rank 1"} (TensorFold)";
      # not started at boot: `systemctl start dsv41-tensorfold`
      wantedBy = lib.mkForce [ ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ] ++ optional cfg.memguard.enable "dsv41-tensorfold-memguard.service";
      # a nixos-rebuild switch must not restart a server that holds ~100 GiB and takes minutes to load
      restartIfChanged = false;
      stopIfChanged = false;
      path = with pkgs; [ iproute2 python3 coreutils gawk gnused gnugrep curl ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [ "video" "render" ];
        ExecStartPre = optional (cfg.minAvailGiB > 0) memoryCheck;
        ExecStart = if isHead then headStart else workerStart;
        ExecStartPost = optional (isHead && cfg.head.waitReady) headReady;
        TimeoutStartSec = if isHead && cfg.head.waitReady then cfg.head.readyTimeout else cfg.minAvailWaitSeconds + 30;
        TimeoutStopSec = 30;
        KillMode = "control-group";
        Restart = "no";
        RuntimeDirectory = "dsv41-tensorfold";
        # what docker gave the ranks: --ulimit memlock=-1 (RDMA registers the buffers), --ulimit stack=67108864
        LimitMEMLOCK = "infinity";
        LimitSTACK = "64M";
        LimitNOFILE = 1048576;
      };
    };

    systemd.services.dsv41-tensorfold-memguard = mkIf cfg.memguard.enable {
      description = "Memory guard of dsv41-tensorfold: kills the rank before the Spark runs out of memory";
      bindsTo = [ "dsv41-tensorfold.service" ];
      partOf = [ "dsv41-tensorfold.service" ];
      restartIfChanged = false;
      path = with pkgs; [ coreutils gawk gnused procps systemd ];
      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = memguardStart;
        RuntimeDirectory = "dsv41-tensorfold-memguard";
        LogsDirectory = "dsv41-tensorfold";
        Restart = "no";
      };
    };
  };
}
