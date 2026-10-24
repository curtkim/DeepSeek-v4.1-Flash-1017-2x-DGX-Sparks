# docker 없이: nix와 `bin/`

`./start.sh`와 같은 Spark 두 대 구성의 서버입니다. 차이는 두 가지입니다.
- 엔진을 `zig/Dockerfile` 대신 `nix build`로 빌드합니다.
- rank를 컨테이너가 아닌 일반 프로세스로 실행합니다.

설정, 링크 감지, 메모리 가드, 각종 검사는 레시피의 것을 그대로 씁니다(`scripts/config.sh`, `scripts/nodes.sh`,
`scripts/memguard.sh`). docker가 하던 일만 대체했습니다.

| | docker (`./start.sh`) | nix (`bin/start`) |
| --- | --- | --- |
| 엔진 | `nvcr.io/nvidia/pytorch:26.07-py3` 위의 `zig/Dockerfile` | `flake.nix`: nixpkgs의 Zig 0.17.0, CUDA 13 (nvcc, cuBLAS), NCCL 2.30.7, rdma-core, Python 3 + Pillow |
| 버전 고정 | `scripts/config.sh`의 `TF_REF`, `INPUTS_REF`, `KIT_REF`, `MODEL_REVISION` | `flake.lock` (입력 `tensorfold`, `tensorfold-inputs`, `kit`, `model`) |
| 워커로 전달 | `docker save \| docker load` | `nix copy` (두 노드에서 같은 `/nix/store` 경로) |
| 워커의 파일 | NFS로 읽음 (docker volume, root 불필요) | CX7 링크를 통해 rsync로 워커 디스크에 복사 (export나 mount 불필요). `WORKER_HF_MOUNT`를 지정했을 때만 NFS |
| rank | 컨테이너 `--ulimit memlock=-1 --ulimit stack=64M` | `setsid` 프로세스. 호스트에서 memlock 한도를 올려야 함 |
| 준비 도구 | 이미지의 Python | `nix build .#tools`: `hf`, torch(CPU)가 든 Python, safetensors, tokenizers, Pillow |
| 로그 | `docker logs` | `~/.cache/dsv41-tensorfold/logs/rank{0,1}-current.log` |


## tensorfold overlay
- ashhart/TensorFold
- bertholomus/TensorFold/tree/deepseek-v41-tp2
- bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10

## nix build
```sh
› t result
result
├── bin
│   ├── tensorfold
│   ├── tensorfold-native
│   ├── tf-dsv41-check
│   ├── tf-dsv41-comm
│   ├── tf-dsv41-engram-check
│   ├── tf-dsv41-lanes
│   ├── tf-dsv41-load
│   ├── tf-dsv41-picture
│   ├── tf-dsv41-rank-cache
│   └── tf-dsv41-rdma-bench
└── share
    └── tensorfold
        ├── pins.env
        └── tools
            └── inputs
                ├── make_bias_vl.py
                ├── make_rope.py
                └── make_token_map.py

```
## flake에 들어 있는 것

```
inputs.tensorfold         BertholomusAI의 TensorFold, branch deepseek-v41-zig @ 629c524a (레시피의 고정 커밋, zig/patches가 여기에 적용됨)
inputs.tensorfold-inputs  같은 fork의 deepseek-v41-zig-tp4 @ 59402086 (tf-dsv41-rank-cache와 token-map / RoPE / image-bias 스크립트)
inputs.kit                이 모델 계열의 커널 kit @ abe07c7d
inputs.model              Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw @ 45ee5d0f, LFS 객체 제외
```

### inputs.kit
엔진이 실행중에 불러쓰는 GPU 커널 바이너리와 상수 파일 모음
엔진소스에서 빌드되는 fatbit(zig/kernels/cuda)과는 별개
kit의 커널은 원래 python엔진(pytorch, triton)에서 컴파일된 것들
zig 엔진은 이미 컴팡리 된 바이너리를 받아서 씀

- aot : triton 커널을 미리 컴파일한 바이너리, 27M
- cubins : EXL3 양자화 가중치용 행렬 곱 커널(linear, exports), 56M
- vision/torch_fmha_sm120.cubin : 비전 타워?용
- engram.json engram의 해시상수
- rope.json, rope-*.f32 : RoPE 위치인코딩 테이블 4개, 136MB
- vision/gate_bias_vl.safetensors : 이미지 구간의 라우팅 bias(Deepseek 가중치에서 꺼낸 작은 텐서 43개)
- MANIFEST : 모든 파일의 sha256


### `inputs.model`

Hugging Face 저장소의 git 트리입니다. config, tokenizer, index, chat template, 그리고 각 shard의 LFS 포인터(sha256과
크기)가 들어 있고, 크기는 약 7 MB입니다.

196 GiB의 가중치는 nix store에 넣지 않습니다. `bin/prepare`가 이 revision의 가중치를 `hf`로 받은 뒤, 모든 파일을 포인터와
대조합니다. 크기는 항상 확인하고, sha256은 `VERIFY_SHA=1`일 때만 확인합니다.

### 모델 저장소 squash와 revision

모델 저장소는 2026-10-19에 squash되었습니다. 그래서 `scripts/config.sh`의 `MODEL_REVISION`(`6d54be4f`)은 이제 Hub에
없습니다. flake는 현재 `main`을 고정했고, 그 shard들의 sha256은 squash된 커밋의 것과 같습니다.

- **캐시에 `6d54be4f`가 이미 있는 경우**: 계속 쓰려면 `scripts/local.sh`에
  `MODEL_REVISION=6d54be4fca338d79b17c3b03cc9a423bb3ac6718`을 지정하세요.
- **새 revision으로 옮기는 경우**: `hf`는 blob을 해시로 재사용하므로 다시 받는 것은 없습니다. 다만 weight 파일 두 개는
  폴더 이름에 revision이 들어가서 새로 만듭니다.

### 출력

- `packages.<system>.{tensorfold (default), tools, zig, nccl, rsync, model-meta, kit}`
- `devShells.<system>.default`
- `nixosModules.default`

`aarch64-linux`(Spark)와 `x86_64-linux`용입니다.

### 엔진 래퍼

엔진 바이너리는 `result/bin/`에 있습니다: `tensorfold-native`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache`와 이 모델 계열의
검사 도구들. 모두 래퍼 스크립트이고, 각 래퍼는 다음 환경을 만듭니다.
- **`LD_LIBRARY_PATH`**: 패키지의 cuBLAS, NCCL, rdma-core를 먼저 넣고, 그 뒤에 호스트의 NVIDIA 드라이버를 넣습니다.
  - 드라이버는 `libcuda.so.1`과 `libnvidia-*` 라이브러리만 `~/.cache/tensorfold-driver`에 링크합니다.
  - 드라이버 폴더는 `TF_DRIVER_LIB_DIR` → `/run/opengl-driver/lib` → `/usr/lib/aarch64-linux-gnu` 등 `libcuda.so.1`이
    있는 곳 순서로 찾습니다.
- **`PATH`**: PNG가 아닌 이미지를 처리할 때 엔진이 띄우는 Pillow가 든 Python을 넣습니다.

## 처음 한 번, 두 Spark 모두 (root)

1. **Nix와 flakes**: 멀티 유저로 설치합니다(https://nixos.org/download).
   - 스크립트가 `--extra-experimental-features "nix-command flakes"`를 스스로 넘기므로 따로 켤 필요는 없습니다.
2. **locked memory 무제한**: RDMA가 버퍼를 등록하기 때문에 필요합니다. docker는 rank에 `memlock=-1`을 주었습니다.
   ```
   echo "$USER - memlock unlimited" | sudo tee /etc/security/limits.d/90-dsv41.conf
   ```
   설정한 뒤 다시 로그인하세요(ssh도 마찬가지).
3. **워커가 head로부터 엔진을 받을 수 있게** 합니다(`nix copy`).
   ```
   # 워커의 /etc/nix/nix.conf
   trusted-users = root <your user>
   ```
   그다음 `sudo systemctl restart nix-daemon`을 실행합니다. 이 설정이 없으면 `bin/prepare`가 워커에서 같은 flake를
   대신 빌드하고, 워커에서 10분쯤 더 걸립니다.
4. **head에서 워커로 키 기반 ssh**:
   - docker 때처럼 `ssh-copy-id`로 키를 등록하고, `scripts/local.sh`에 `WORKER=user@<worker>`를 지정합니다.
   - rsync는 CX7 링크로 복사하므로, 워커는 **CX7 링크 주소로 접속할 때도** 같은 키를 받아야 합니다.
   - 두 접속 모두 `bin/check`가 확인합니다.

## 워커의 파일: 복사(기본) 또는 NFS

rank 1이 읽는 것은 다음과 같습니다(`zig/src/families/dsv41/model.zig`).
- 자신의 weight 파일
- Engram 테이블
- kit
- token map
- checkpoint에서는 `config.json` 하나뿐

checkpoint의 shard를 여는 건 rank 0뿐입니다(비전 타워). 그래서 기본적으로 `bin/prepare`가 워커에 자체 사본을 만들어
주고, NFS는 필요 없습니다.

| 워커의 위치 | 내용 | 크기 |
| --- | --- | --- |
| `WORKER_DATA_DIR` (`~/.cache/dsv41-tensorfold/data-<rev>`) | shard를 뺀 checkpoint 파일들, kit, token map | 약 100 MB |
| `WORKER_ENGRAM_DIR` | Engram 테이블 두 개 | 약 203 GB |
| `WORKER_CACHE_DIR` | rank 1의 weight 파일 | 약 106 GB |

복사는 이렇게 동작합니다.
- rsync는 워커의 CX7 링크 주소로 갑니다(`WORKER_SYNC_HOST`로 바꿀 수 있음). 양쪽 모두 flake의 rsync를 씁니다.
- 중간에 끊긴 복사는 다음 실행에서 이어받습니다.
- rank 1의 weight 파일은 이 Spark에서 먼저 만들기 때문에, prepare 중에는 `HF_CACHE`에 약 115 GB의 여유가 잠시 더
  필요합니다. 워커가 파일을 전부 받으면 로컬 사본은 지웁니다.

NFS보다 링크로 오가는 양이 적고(196 GiB checkpoint를 읽는 대신 106 GB), 어느 Spark에도 export나 mount가 필요 없습니다.

### NFS를 쓰려면

`WORKER_HF_MOUNT`에 워커가 이 Spark의 `HF_CACHE`를 마운트한 위치를 지정합니다. 그러면 docker 레시피처럼, 워커가
그 마운트에서 checkpoint, kit, token map을 읽고 자신의 weight 파일도 거기서 만듭니다. 두 Spark 모두 root가 필요합니다.

- **head**: `HF_CACHE`를 워커의 링크 주소에 읽기 전용으로 export합니다.
  ```
  # /etc/exports
  /home/<you>/.cache/huggingface  <worker link address>(ro,no_subtree_check,no_root_squash)
  ```
  그다음 `sudo exportfs -ra`를 실행합니다.
- **워커**: 마운트합니다. 가능하면 같은 경로에 두세요.
  ```
  # /etc/fstab
  <head link address>:/home/<you>/.cache/huggingface  /home/<you>/.cache/huggingface  nfs4  ro,nconnect=8,rsize=1048576,hard,timeo=600,_netdev  0 0
  ```
  그다음 그 경로에 `sudo mount`를 실행합니다.

## 실행

```
bin/check            # 두 Spark: nix, 드라이버, memlock, RoCE, 링크 ssh(또는 NFS), 메모리, 디스크, 엔진. FAIL마다 고치는 방법 표시
bin/prepare          # nix build + nix copy, checkpoint(196 GiB)와 Engram(203 GB), kit, weight 파일 두 개, 워커 사본
bin/start            # 필요하면 bin/prepare를 실행한 뒤, 워커의 rank 1, 이 Spark의 rank 0, smoke test
bin/start restart    # 설정을 바꾼 뒤 (CONTEXT=262144 bin/start restart ...)
bin/stop
```

명령들은 `./start.sh`와 같은 설정과 스위치를 받습니다: `DRY_RUN=1`, `FOREGROUND=1`, `API_KEY=1`, `ABLIT=1`,
`PREPARE=0|1`, `MEMGUARD_GIB`, `NCCL_*`, `TF_DS_*`.

nix 쪽 명령은 상태를 따로 둡니다.
- 마커: `~/.local/state/dsv41-tensorfold/prepared-nix`
- out-link: 같은 폴더의 `engine`, `tools`, `rsync`, `model-meta`. store 경로가 가비지 컬렉션되지 않게 막아 줍니다.

그래서 docker 레시피와 nix 레시피는 같은 Spark에 함께 둘 수 있습니다. 다만 같은 포트를 쓰므로 한 번에 하나만 실행할
수 있습니다.

### TensorFold 버전 올리기

1. `flake.nix`에서 `inputs.tensorfold`의 `rev`를 바꿉니다(`ref=deepseek-v41-zig`는 그대로 둠).
2. `nix flake lock`을 실행합니다.

먼저 `zig/patches`가 새 버전에도 적용되는지 확인하세요. 적용되지 않는 패치가 있으면 `nix build`가 거기서 멈춥니다.

## NixOS: `services.dsv41-tensorfold`

NixOS를 쓰는 Spark에서는 flake의 `nixosModules.default`가 `bin/start` 대신 각 rank를 `dsv41-tensorfold.service`로 띄웁니다.
- 유닛은 **부팅 때 시작되지 않습니다**. 실행 중에는 `nixos-rebuild switch`가 재시작하지 않습니다.
- memlock과 스택 한도는 유닛에 들어 있으므로(`LimitMEMLOCK=infinity`, `LimitSTACK=64M`) `limits.d`가 필요 없습니다.
- rdma-core의 udev 규칙과 RoCE 커널 모듈도 모듈이 넣습니다.
- 메모리 가드는 각 rank 옆에서 `dsv41-tensorfold-memguard.service`로 돕니다.

데이터는 여전히 `bin/prepare`가 준비합니다. head에서 서비스와 같은 사용자로 한 번 실행하면, 모듈의 기본 경로가 그
배치와 맞습니다.

```nix
# 시스템 설정의 flake.nix
inputs.dsv41.url = "github:<you>/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks";   # 또는 path:/home/curt/projects/...

# 두 Spark 공통 설정 (두 rank가 같은 값을 써야 함)
{ inputs, ... }: {
  imports = [ inputs.dsv41.nixosModules.default ];
  # NVIDIA 드라이버를 설정해 두어야 합니다(hardware.nvidia / hardware.graphics). 엔진은 libcuda.so.1을
  # /run/opengl-driver/lib 또는 services.dsv41-tensorfold.driverLibDir에서 찾습니다
  services.dsv41-tensorfold = {
    enable = true;
    user = "curt";                        # bin/prepare를 실행한 사용자
    context = 1048576;
    parallel = 4;
    network.trustedInterfaces = [ "enp1s0f1np1" "enP2p1s0f1np1" ];   # CX7 링크 (rendezvous와 NCCL 포트)
  };
}

# head
services.dsv41-tensorfold = {
  role = "head";
  peerAddress = "192.168.100.11";         # 워커의 CX7 링크 주소
  openFirewall = true;                    # API 포트 8888
  # head.apiKeyFile = "/run/secrets/dsv41-api-key";
};

# 워커
services.dsv41-tensorfold = {
  role = "worker";
  peerAddress = "192.168.100.10";         # head의 CX7 링크 주소 (rank 0의 rendezvous)
};
```

```
worker$ sudo systemctl start dsv41-tensorfold    # 먼저: rank 1이 rank 0의 rendezvous를 기다림
head$   sudo systemctl start dsv41-tensorfold    # API가 응답하면 반환 (head.waitReady, 최대 40분)
        journalctl -fu dsv41-tensorfold          # rank의 로그
        sudo systemctl stop dsv41-tensorfold     # 양쪽 모두: 혼자 남은 rank는 상대를 계속 기다림
```

### 알아 둘 옵션

- **서빙 설정**: `scripts/config.sh`의 이름을 camelCase로 씁니다.
  - `context`, `parallel`, `roundRows`, `maxTokens`, `thinking`, `reasoningEffort`, `thinkingBudget`
  - `temperature`, `topP`, `topK`
  - `drafts`, `rdma`, `graphs`, `side`, `prefetch`, `engramAio`, `replayFloor`, `head.prefillInterleaveMs`
- **환경 변수와 인자**: `environment`로 `TF_DS_*`, `NCCL_*`를 더하고, `head.extraArgs`, `worker.extraArgs`로 명령줄 인자를
  더합니다.
- **`network.rails = 1`**: 케이블이 꽂힌 포트의 장치만 씁니다. PCIe twin이 레인마다 한 서브넷에 같은 MTU로 주소가 잡혀
  있지 않을 때 쓰세요. `bin/start`는 이를 미리 확인하지만 유닛은 확인하지 않습니다.
- **`network.hcas`, `network.gidIndex`, `network.socketIfname`**: 링크 감지 결과를 덮어씁니다.
- **`head.*`, `worker.*` 경로**: `dataDir`, `servedDir`, `rankCacheDir`, `kitDir`, `engramDir`, `tokenMap`, `modelDir`.
  `bin/prepare` 기본 배치와 다를 때 씁니다. 예: `ABLIT=1`이면 `head.dataDir = ".../<rev>-ablit-<rev>"`.
- **그 밖**: `minAvailGiB`, `memguard.{enable,floorGiB,holdSeconds,hardFloorGiB}`, `driverLibDir`, `package`.
