# Shared by bin/prepare, bin/check, bin/start and bin/stop: the recipe without docker. The settings are scripts/config.sh's
# (environment > scripts/local.sh > ./.env > defaults) and the two Sparks' link is scripts/nodes.sh's; what docker did
# is done here instead:
#   - the engine is the flake's package (`nix build`, flake.nix), the same /nix/store path on both Sparks: built here,
#     copied to the worker with `nix copy` (or built there from the same flake when the worker does not take unsigned
#     paths from this Spark), kept from garbage collection by an out-link on each;
#   - the ranks are processes in their own session (setsid), their pid in a file and their output in a log file;
#   - the checkpoint is read at its host path (HF_CACHE) here. The worker gets its own copy of what rank 1 reads (rsync
#     over the link: its weight file, made here, Engram's tables, the kit, the token map and the checkpoint's small
#     files), or, when WORKER_HF_MOUNT is set, reads this Spark's HF_CACHE where it mounts it over NFS.
# The pins (TensorFold's branch and commit, the inputs' tools, the kit, the checkpoint's revision) are flake.lock's.
_bin_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$_bin_root" || exit 1

NIX=(nix --extra-experimental-features "nix-command flakes")
# lock_rev <input>: the commit flake.lock pins for a flake input
lock_rev() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["nodes"][sys.argv[2]]["locked"]["rev"])' "$_bin_root/flake.lock" "$1"; }
[[ -f flake.lock ]] || { echo "flake.lock is missing: run nix flake lock in $_bin_root" >&2; exit 1; }
# The engine's pins come from the flake (config.sh's TF_REF / INPUTS_REF / KIT_REF are the docker image's)
export TF_REF=$(lock_rev tensorfold) INPUTS_REF=$(lock_rev tensorfold-inputs) KIT_REF=$(lock_rev kit)
# The checkpoint's revision: the flake's, unless you set MODEL_ID or MODEL_REVISION yourself (environment, scripts/local.sh
# or .env), e.g. to a snapshot already in your cache under another revision
if [[ -z "${MODEL_REVISION+x}" && -z "${MODEL_ID+x}" ]] && ! grep -qsE '^[[:space:]]*(export[[:space:]]+)?(MODEL_REVISION|MODEL_ID)=' scripts/local.sh .env; then
  export MODEL_REVISION=$(lock_rev model)
  FLAKE_MODEL=1
else
  FLAKE_MODEL=0
fi
source ./scripts/config.sh
source ./scripts/nodes.sh
# config.sh's model_in_container / served_in_container are /hf paths; here they are host paths (below)
[[ "$MODEL_REVISION" == "$(lock_rev model)" && "$MODEL_ID" == Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw ]] && FLAKE_MODEL=1

# ---------------------------------------------------------------- where things are
# How the worker gets its files: copied to its own disk (the default; no root needed anywhere), or this Spark's HF_CACHE
# read over NFS when WORKER_HF_MOUNT names where the worker mounts it
WORKER_HF_MOUNT="${WORKER_HF_MOUNT:-}"
if [[ -n "$WORKER_HF_MOUNT" ]]; then WORKER_TRANSFER=nfs; else WORKER_TRANSFER=copy; fi
# copy: the worker's own copy of what rank 1 reads besides its weight file (WORKER_CACHE_DIR) and Engram's tables
# (WORKER_ENGRAM_DIR): the checkpoint's small files (its config; no shards), the kit and the token map
WORKER_DATA_DIR="${WORKER_DATA_DIR:-\$HOME/.cache/dsv41-tensorfold/data-${DATA_SUB##*/}}"
# copy: the address rsync reaches the worker at (default: the user of WORKER at the worker's address on the CX7 link)
WORKER_SYNC_HOST="${WORKER_SYNC_HOST:-}"
WSTATE='$HOME/.cache/dsv41-tensorfold'                    # on the worker (expanded there)
RANK0_PID="$STATE_DIR/rank0.pid"
RANK0_LOG="$LOG_DIR/rank0-current.log"
RANK1_PID="$WSTATE/rank1.pid"
RANK1_LOG="$WSTATE/logs/rank1-current.log"
RUN_STATE="$STATE_DIR/running"                            # what the running ranks were started with
ENGINE_LINK="$STATE_DIR/engine"                           # out-links: keep the store paths from garbage collection
TOOLS_LINK="$STATE_DIR/tools"
RSYNC_LINK="$STATE_DIR/rsync"

# a path under HF_CACHE as the worker sees it over NFS
wpath() { [[ "$1" == "$HF_CACHE"* ]] || { echo "$1"; return; }; echo "$WORKER_HF_MOUNT${1#"$HF_CACHE"}"; }
model_dir()   { if [[ "$ABLIT" == 1 ]]; then echo "$DATA_DIR/checkpoint"; else echo "$(model_cache_dir)/snapshots/$(snapshot_rev)"; fi; }
engram_dir()  { echo "$(engram_cache_dir)/snapshots/$(engram_rev)"; }
served_dir()  { echo "$DATA_DIR/model"; }
kit_dir()     { echo "$HF_CACHE/$KIT_SUB"; }
# what rank 1 is given on the worker (its paths there; a $HOME in them is expanded there)
w_model_dir() { if [[ $WORKER_TRANSFER == nfs ]]; then wpath "$(model_dir)"; else echo "$WORKER_DATA_DIR/model"; fi; }
w_kit_dir()   { if [[ $WORKER_TRANSFER == nfs ]]; then wpath "$(kit_dir)/kit"; else echo "$WORKER_DATA_DIR/kit"; fi; }
w_token_map() { if [[ $WORKER_TRANSFER == nfs ]]; then echo "$(wpath "$DATA_DIR")/dsv41_token_map.json"; else echo "$WORKER_DATA_DIR/dsv41_token_map.json"; fi; }
# wabs <path on the worker>: the path with the worker's $HOME expanded (rsync's destination is taken literally)
wabs() { worker 1 "echo $1"; }

# ---------------------------------------------------------------- nix
# engine_build: the flake's engine and prepare tools here (ENGINE, TOOLS), and PATH with the tools' hf and python
engine_build() {
  mkdir -p "$STATE_DIR"
  log "Building the engine (nix build .#tensorfold: TensorFold ${TF_REF:0:8} + $(compgen -G 'zig/patches/*.patch' | wc -l) patches, inputs' tools ${INPUTS_REF:0:8}; the first time ~10 min, NCCL included)"
  "${NIX[@]}" build "$_bin_root#tensorfold" --out-link "$ENGINE_LINK" || die "nix build .#tensorfold failed"
  "${NIX[@]}" build "$_bin_root#tools" --out-link "$TOOLS_LINK" || die "nix build .#tools failed"
  "${NIX[@]}" build "$_bin_root#rsync" --out-link "$RSYNC_LINK" || die "nix build .#rsync failed"
  engine_paths
}
# engine_paths: ENGINE and TOOLS from the out-links an earlier engine_build left (no build)
engine_paths() {
  ENGINE=$(readlink -f "$ENGINE_LINK" 2>/dev/null || true)
  TOOLS=$(readlink -f "$TOOLS_LINK" 2>/dev/null || true)
  RSYNC=$(readlink -f "$RSYNC_LINK" 2>/dev/null || true)
  [[ -n "$TOOLS" && -x "$TOOLS/bin/python3" ]] && export PATH="$TOOLS/bin:$PATH"
  [[ -n "$ENGINE" && -x "$ENGINE/bin/tensorfold-native" ]]
}
# the store path the flake's engine has now (evaluation only): differs from ENGINE when flake.nix, flake.lock or zig/patches changed
engine_wanted() { "${NIX[@]}" eval --raw "$_bin_root#tensorfold.outPath" 2>/dev/null; }

# The worker's nix (a non-interactive ssh does not read the profile that puts it in PATH)
WORKER_NIX_BIN="${WORKER_NIX_BIN:-}"
worker_nix_bin() {
  [[ -n "$WORKER_NIX_BIN" ]] && { echo "$WORKER_NIX_BIN"; return 0; }
  WORKER_NIX_BIN=$(worker 1 'for d in /run/current-system/sw/bin /nix/var/nix/profiles/default/bin "$HOME/.nix-profile/bin"; do [ -x "$d/nix" ] && { echo "$d"; exit 0; }; done; d=$(command -v nix) && dirname "$d"' 2>/dev/null) || true
  [[ -n "$WORKER_NIX_BIN" ]] || return 1
  echo "$WORKER_NIX_BIN"
}
# engine_deploy: the same store paths on the worker (the engine, and the rsync bin/prepare copies with), kept by out-links
# there ($WSTATE/engine, $WSTATE/rsync)
engine_deploy() {
  local nb store
  nb=$(worker_nix_bin) || die "$(worker_host 1) has no nix (install it there, or set WORKER_NIX_BIN to the folder that holds nix)"
  store="ssh-ng://$(worker_host 1)?remote-program=$nb/nix-daemon"
  if worker 1 "[ -x '$ENGINE/bin/tensorfold-native' ] && [ -x '$RSYNC/bin/rsync' ]"; then
    log "Worker: the engine is there already ($ENGINE)"
  else
    log "Copying the engine to $(worker_host 1) (nix copy; only what it lacks)"
    if ! "${NIX[@]}" copy --to "$store" "$ENGINE" "$RSYNC" 2>"$STATE_DIR/nix-copy.err"; then
      warn "nix copy to $(worker_host 1) was refused ($(tail -1 "$STATE_DIR/nix-copy.err")): building the same flake there instead. (To copy instead, add your user to trusted-users in the worker's nix.conf)"
      local src
      src=$("${NIX[@]}" flake archive --json "$_bin_root" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["path"])') ||
        die "nix flake archive failed"
      "${NIX[@]}" flake archive --to "$store" "$_bin_root" || die "could not copy the flake's sources to $(worker_host 1)"
      worker 1 "$nb/nix --extra-experimental-features 'nix-command flakes' build '$src#tensorfold' '$src#rsync' --no-link" ||
        die "nix build of the engine failed on $(worker_host 1)"
    fi
    worker 1 "[ -x '$ENGINE/bin/tensorfold-native' ] && [ -x '$RSYNC/bin/rsync' ]" ||
      die "$(worker_host 1) built a different engine path than $ENGINE: both Sparks need the same nixpkgs and system (aarch64-linux)"
  fi
  worker 1 "mkdir -p $WSTATE && $nb/nix --extra-experimental-features nix-command build '$ENGINE' --out-link $WSTATE/engine && $nb/nix --extra-experimental-features nix-command build '$RSYNC' --out-link $WSTATE/rsync" >/dev/null ||
    warn "could not add a garbage-collector root for the engine on $(worker_host 1)"
}

# wsync <destination on the worker (absolute: wabs)> <rsync options and sources...>: copy to the worker's disk over the
# CX7 link, with the flake's rsync on both ends; links in the sources are followed (a snapshot's files are links into
# blobs/); a broken copy is kept (--partial) and the big files resume from it (--append-verify, which rules out -W: a file
# the worker does not have is sent whole anyway)
wsync_host() {
  [[ -n "$WORKER_SYNC_HOST" ]] && { echo "$WORKER_SYNC_HOST"; return; }
  local user=""; [[ "$WORKER" == *@* ]] && user="${WORKER%@*}@"
  echo "$user${LINK_WORKER_ADDR[1]:-${WORKER#*@}}"
}
wsync() {
  local dest=$1; shift
  "$RSYNC/bin/rsync" -a --partial --copy-links --info=progress2 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o Compression=no" \
    --rsync-path "$RSYNC/bin/rsync" "$@" "$(wsync_host):$dest"
}

# ---------------------------------------------------------------- the ranks
# rank_alive_here / rank_alive_worker: the pid in the rank's pid file runs one of the engine's binaries
rank_pid_here() { local p; p=$(cat "$RANK0_PID" 2>/dev/null) && [[ "$p" =~ ^[0-9]+$ ]] && grep -qa tensorfold "/proc/$p/cmdline" 2>/dev/null && echo "$p"; }
rank_alive_here() { [[ -n "$(rank_pid_here)" ]]; }
rank_alive_worker() {
  worker 1 "p=\$(cat $RANK1_PID 2>/dev/null) && [ -n \"\$p\" ] && grep -qa tensorfold /proc/\$p/cmdline 2>/dev/null"
}
# nccl_assign <rank>: the rank's NCCL settings as VAR=value words (scripts/nodes.sh's, without docker's -e)
nccl_assign() {
  rank_nccl_args "$1"
  local a out=()
  for a in "${NCCL_ARGS[@]}"; do [[ "$a" == -e ]] && continue; out+=("$a"); done
  printf '%s\n' "${out[@]}"
}
# the limits docker gave the ranks: locked memory unlimited (RDMA registers the weights' and caches' buffers) and a 64 MiB stack
RANK_LIMITS='ulimit -l unlimited 2>/dev/null || ulimit -l "$(ulimit -H -l)"; ulimit -s 65536 2>/dev/null || true'

# log_rotate <dir> <rank> <current log> <keep>: the current log gzipped as <date>-<time>-rank<N>.log.gz, the newest <keep>
# kept (scripts/nodes.sh's save_log, for a file). Self-contained: it runs on the worker too.
log_rotate() {
  local dir=$1 rank=$2 cur=$3 keep=$4 stamp f i=0
  [[ -s "$cur" ]] || return 0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || { rm -f "$cur"; return 0; }
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done
  gzip -c "$cur" > "$f" && rm -f "$cur" || { rm -f "$f"; return 1; }
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}

# stop_group <pid> <seconds>: SIGTERM to the process group, SIGKILL after <seconds>. Self-contained (runs on the worker).
stop_group() {
  local pid=$1 t=$2 n=0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || return 0
  while [[ -d /proc/$pid ]] && (( n < t * 5 )); do sleep 0.2; n=$((n + 1)); done
  [[ -d /proc/$pid ]] && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; sleep 0.5; }
  return 0
}

# The memory guard (scripts/memguard.sh) on both Sparks, killing a rank's process group instead of a container
memguard_start() {
  awk -v g="$MEMGUARD_GIB" 'BEGIN {exit !(g > 0)}' || return 0
  "$_bin_root/scripts/memguard.sh" start "pidfile:$RANK0_PID" "$MEMGUARD_GIB" "$LOG_DIR/memguard.log" "$STATE_DIR/memguard.pid" "$MEMGUARD_HOLD_S" "$MEMGUARD_HARD_GIB" ||
    die "the memory guard did not start here"
  worker 1 "mkdir -p $WSTATE/logs && cat > $WSTATE/memguard.sh && chmod +x $WSTATE/memguard.sh" < "$_bin_root/scripts/memguard.sh" ||
    die "could not copy the memory guard to $(worker_host 1)"
  worker 1 "$WSTATE/memguard.sh start pidfile:$RANK1_PID $MEMGUARD_GIB $WSTATE/logs/memguard.log $WSTATE/memguard.pid $MEMGUARD_HOLD_S $MEMGUARD_HARD_GIB" ||
    die "the memory guard did not start on $(worker_host 1)"
  log "Memory guard on: a rank is killed when the memory available stays under $MEMGUARD_GIB GiB for $MEMGUARD_HOLD_S s, or falls under $MEMGUARD_HARD_GIB GiB (log: memguard.log)"
}
memguard_stop() {
  "$_bin_root/scripts/memguard.sh" stop "$STATE_DIR/memguard.pid" 2>/dev/null || true
  worker 1 "[ -x $WSTATE/memguard.sh ] && $WSTATE/memguard.sh stop $WSTATE/memguard.pid; true" 2>/dev/null || true
}

# What prepare leaves ready (bin/start runs bin/prepare again when this changes): config.sh's, with the engine's store path
# in place of the image
prepared_state() {
  echo "model=$MODEL_ID@$(snapshot_rev) engram=$ENGRAM_ID@$(engram_rev) engine=$(engine_wanted) kit=$KIT_REF" \
       "w=$(worker_host 1) transfer=$WORKER_TRANSFER wmount=$WORKER_HF_MOUNT wdata=$WORKER_DATA_DIR wengram=$WORKER_ENGRAM_DIR wcache=$WORKER_CACHE_DIR ablit=$ABLIT" \
       "overlay=$( [[ "$ABLIT" == 1 ]] && echo "$ABLIT_ID@$ABLIT_REVISION+stock-dtypes" || echo none)"
}
PREPARED_MARKER="$STATE_DIR/prepared-nix"   # not the docker recipe's marker: the two prepare different things

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
