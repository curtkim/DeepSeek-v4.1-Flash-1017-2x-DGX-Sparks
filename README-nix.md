# Without docker: nix and `bin/`

The same two-Spark server as `./start.sh`, with the engine built by `nix build` instead of `zig/Dockerfile` and the ranks
run as plain processes instead of containers. Settings, the link detection, the memory guard and the checks are the
recipe's own (`scripts/config.sh`, `scripts/nodes.sh`, `scripts/memguard.sh`); only what docker did is replaced.

| | docker (`./start.sh`) | nix (`bin/start`) |
| --- | --- | --- |
| engine | `zig/Dockerfile` on `nvcr.io/nvidia/pytorch:26.07-py3` | `flake.nix`: Zig 0.17.0, CUDA 13 (nvcc, cuBLAS), NCCL 2.30.7, rdma-core, Python 3 + Pillow from nixpkgs |
| pins | `TF_REF`, `INPUTS_REF`, `KIT_REF`, `MODEL_REVISION` in `scripts/config.sh` | `flake.lock` (inputs `tensorfold`, `tensorfold-inputs`, `kit`, `model`) |
| to the worker | `docker save \| docker load` | `nix copy` (the same `/nix/store` path on both) |
| the worker's files | read over NFS (a docker volume, no root) | copied to its disk with rsync over the CX7 link (no exports or mounts); NFS only if you set `WORKER_HF_MOUNT` |
| ranks | containers `--ulimit memlock=-1 --ulimit stack=64M` | `setsid` processes; the memlock limit must be raised on the host |
| prepare tools | the image's Python | `nix build .#tools`: `hf`, Python with torch (CPU), safetensors, tokenizers, Pillow |
| logs | `docker logs` | `~/.cache/dsv41-tensorfold/logs/rank{0,1}-current.log` |

## What the flake holds

```
inputs.tensorfold         BertholomusAI's TensorFold, branch deepseek-v41-zig @ 629c524a (the recipe's pin; zig/patches apply to it)
inputs.tensorfold-inputs  the same fork's deepseek-v41-zig-tp4 @ 59402086 (tf-dsv41-rank-cache and the token-map / RoPE / image-bias scripts)
inputs.kit                the family's kernel kit @ abe07c7d
inputs.model              Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw @ 45ee5d0f, without its LFS objects
```

`inputs.model` is the Hub repository's git tree: config, tokenizer, index, chat template and each shard's LFS pointer
(sha256 and size), ~7 MB. The 196 GiB of weights are not in the nix store: `bin/prepare` downloads them with `hf` at this
revision and checks every file against the pointers (sizes always, sha256 with `VERIFY_SHA=1`).

The model repository was squashed on 2026-10-19, so `scripts/config.sh`'s `MODEL_REVISION` (`6d54be4f`) no longer exists on
the Hub. The flake pins today's `main`, whose shards have the same sha256 as the squashed commit's. If your cache already
holds `6d54be4f`, set `MODEL_REVISION=6d54be4fca338d79b17c3b03cc9a423bb3ac6718` in `scripts/local.sh` to keep using it.
`hf` also reuses blobs by hash, so moving to the new revision downloads nothing again; only the two weight files are
rewritten, because their folders are named after the revision.

Outputs: `packages.<system>.{tensorfold (default), tools, zig, nccl, model-meta, kit}` and `devShells.<system>.default`,
for `aarch64-linux` (the Sparks) and `x86_64-linux`.

The engine's binaries are in `result/bin/`: `tensorfold-native`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache`, and the family's
check tools. Each one is a wrapper:
- `LD_LIBRARY_PATH` gets the package's cuBLAS, NCCL and rdma-core.
- After those comes the host's NVIDIA driver. Only `libcuda.so.1` and the `libnvidia-*` libraries are linked, into
  `~/.cache/tensorfold-driver`. The driver folder is found from `TF_DRIVER_LIB_DIR`, else `/run/opengl-driver/lib`, else
  wherever `libcuda.so.1` sits under `/usr/lib/aarch64-linux-gnu`.
- `PATH` gets the Python with Pillow that the engine starts for pictures other than PNG.

## Once, on both Sparks (root)

1. **Nix with flakes**, multi-user (https://nixos.org/download). The scripts pass `--extra-experimental-features "nix-command flakes"`
   themselves.
2. **Locked memory unlimited.** Docker gave the ranks `memlock=-1`, and RDMA registers the buffers.
   ```
   echo "$USER - memlock unlimited" | sudo tee /etc/security/limits.d/90-dsv41.conf
   ```
   Then log in again (ssh too).
3. **The worker may receive the engine** from the head (`nix copy`):
   ```
   # /etc/nix/nix.conf on the worker
   trusted-users = root <your user>
   ```
   Then run `sudo systemctl restart nix-daemon`. Without this, `bin/prepare` builds the same flake on the worker instead,
   which takes about 10 more minutes there.
4. **Key-based ssh** from the head to the worker (`ssh-copy-id`), and `WORKER=user@<worker>` in `scripts/local.sh`, as for
   docker. The worker must also take that key at its address on the CX7 link, because rsync copies over the link.
   `bin/check` tests both.

## The worker's files: copied (default) or NFS

Rank 1 reads its weight file, Engram's tables, the kit, the token map, and from the checkpoint only `config.json`
(`zig/src/families/dsv41/model.zig`). Only rank 0 opens the checkpoint's shards (the vision tower). So by default
`bin/prepare` gives the worker its own copy and needs no NFS:

| on the worker | what | how big |
| --- | --- | --- |
| `WORKER_DATA_DIR` (`~/.cache/dsv41-tensorfold/data-<rev>`) | the checkpoint's files but its shards, the kit, the token map | ~100 MB |
| `WORKER_ENGRAM_DIR` | Engram's two tables | ~203 GB |
| `WORKER_CACHE_DIR` | rank 1's weight file | ~106 GB |

The copies work like this:
- rsync goes to the worker's address on the CX7 link (`WORKER_SYNC_HOST` overrides it). Both ends run the flake's rsync.
- An interrupted copy resumes on the next run.
- Rank 1's weight file is made on this Spark first, so prepare needs ~115 GB more free in `HF_CACHE` for a while. The
  local copy is deleted once the worker holds the whole file.

This moves less over the link than NFS did (106 GB instead of reading the 196 GiB checkpoint), and needs no export
or mount on either Spark.

**NFS instead**: set `WORKER_HF_MOUNT` to where the worker mounts this Spark's `HF_CACHE`. The worker then reads the
checkpoint, kit and token map there and makes its own weight file from the mount, like the docker recipe. This needs
root on both Sparks:
- On the head, export `HF_CACHE` read-only to the worker's link address:
  ```
  # /etc/exports
  /home/<you>/.cache/huggingface  <worker link address>(ro,no_subtree_check,no_root_squash)
  ```
  Then run `sudo exportfs -ra`.
- On the worker, mount it, preferably at the same path:
  ```
  # /etc/fstab
  <head link address>:/home/<you>/.cache/huggingface  /home/<you>/.cache/huggingface  nfs4  ro,nconnect=8,rsize=1048576,hard,timeo=600,_netdev  0 0
  ```
  Then run `sudo mount` on that path.

## Run

```
bin/check            # both Sparks: nix, driver, memlock, RoCE, ssh over the link (or NFS), memory, disk, engine; how to fix each FAIL
bin/prepare          # nix build + nix copy, the checkpoint (196 GiB) and Engram (203 GB), the kit, both weight files, the worker's copy
bin/start            # runs bin/prepare when needed, then rank 1 on the worker, rank 0 here, the smoke test
bin/start restart    # after changing a setting (CONTEXT=262144 bin/start restart ...)
bin/stop
```

The commands take the same settings and switches as `./start.sh`: `DRY_RUN=1`, `FOREGROUND=1`, `API_KEY=1`, `ABLIT=1`,
`PREPARE=0|1`, `MEMGUARD_GIB`, `NCCL_*`, `TF_DS_*`. The nix commands keep their own state:
- marker: `~/.local/state/dsv41-tensorfold/prepared-nix`
- out-links: `engine`, `tools`, `rsync` and `model-meta` in that same folder (they protect the store paths from garbage collection)

So the docker recipe and the nix one can both live on the same Sparks, but only one of them may run at a time (they
serve on the same port).

To move to a newer TensorFold, change the `rev` of `inputs.tensorfold` in `flake.nix` (keep `ref=deepseek-v41-zig`) and run
`nix flake lock`. Check that `zig/patches` still apply first: `nix build` stops at the first patch that does not.

## NixOS: `services.dsv41-tensorfold`

On Sparks running NixOS, the flake's `nixosModules.default` runs each rank as `dsv41-tensorfold.service` instead of
`bin/start`:
- The unit is **not started at boot**, and a `nixos-rebuild switch` does not restart it while it runs.
- Memlock and stack limits are the unit's own (`LimitMEMLOCK=infinity`, `LimitSTACK=64M`), so `limits.d` is not needed.
- rdma-core's udev rules and the RoCE kernel modules come with the module.
- The memory guard runs beside each rank as `dsv41-tensorfold-memguard.service`.

The data is still `bin/prepare`'s. Run it once from the head as the same user, and the default paths match its layout.

```nix
# flake.nix of your systems
inputs.dsv41.url = "github:<you>/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks";   # or path:/home/curt/projects/...

# both Sparks: the shared settings (they must be the same on both ranks)
{ inputs, ... }: {
  imports = [ inputs.dsv41.nixosModules.default ];
  # the NVIDIA driver must be set up (hardware.nvidia / hardware.graphics): the engine finds libcuda.so.1 in
  # /run/opengl-driver/lib, or in services.dsv41-tensorfold.driverLibDir
  services.dsv41-tensorfold = {
    enable = true;
    user = "curt";                        # the user who ran bin/prepare
    context = 1048576;
    parallel = 4;
    network.trustedInterfaces = [ "enp1s0f1np1" "enP2p1s0f1np1" ];   # the CX7 link (rendezvous and NCCL ports)
  };
}

# the head
services.dsv41-tensorfold = {
  role = "head";
  peerAddress = "192.168.100.11";         # the worker's address on the CX7 link
  openFirewall = true;                    # the API's port 8888
  # head.apiKeyFile = "/run/secrets/dsv41-api-key";
};

# the worker
services.dsv41-tensorfold = {
  role = "worker";
  peerAddress = "192.168.100.10";         # the head's address on the CX7 link (rank 0's rendezvous)
};
```

```
worker$ sudo systemctl start dsv41-tensorfold    # first: rank 1 waits for rank 0's rendezvous
head$   sudo systemctl start dsv41-tensorfold    # returns once the API answers (head.waitReady; up to 40 min)
        journalctl -fu dsv41-tensorfold          # the rank's log
        sudo systemctl stop dsv41-tensorfold     # on both: a rank left alone waits for its peer
```

Options worth knowing:
- Serving settings use `scripts/config.sh`'s names in camelCase: `context`, `parallel`, `roundRows`, `maxTokens`,
  `thinking`, `reasoningEffort`, `thinkingBudget`, `temperature`, `topP`, `topK`, `drafts`, `rdma`, `graphs`, `side`,
  `prefetch`, `engramAio`, `replayFloor`, `head.prefillInterleaveMs`.
- `environment` adds `TF_DS_*` and `NCCL_*`. `head.extraArgs` and `worker.extraArgs` add command-line arguments.
- `network.rails = 1` uses only the cabled port's device. Use it when the PCIe twins are not addressed on one subnet per
  rail with the same MTU: `bin/start` probes that, the unit does not.
- `network.hcas`, `network.gidIndex` and `network.socketIfname` override the link detection.
- `head.*` and `worker.*` paths (`dataDir`, `servedDir`, `rankCacheDir`, `kitDir`, `engramDir`, `tokenMap`, `modelDir`)
  are for a layout other than `bin/prepare`'s default, e.g. `ABLIT=1` (`head.dataDir = ".../<rev>-ablit-<rev>"`).
- `minAvailGiB`, `memguard.{enable,floorGiB,holdSeconds,hardFloorGiB}`, `driverLibDir`, `package`.
