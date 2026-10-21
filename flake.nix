{
  # The engine and the tools of this recipe without docker: `nix build` compiles BertholomusAI's TensorFold fork (branch
  # deepseek-v41-zig at the recipe's pin, plus zig/patches) with Zig 0.17.0 and nvcc for the GB10's sm_121, the inputs'
  # tools of its four-node branch (tf-dsv41-rank-cache and the token-map / RoPE / image-bias scripts), and wraps every
  # binary with the CUDA 13 libraries, NCCL, rdma-core and the Python with Pillow the engine starts for pictures other
  # than PNG. The host keeps only the NVIDIA driver (libcuda.so.1). bin/prepare, bin/check, bin/start and bin/stop use it.
  #
  #   nix build                  # the engine (result/bin/tensorfold-native, tf-dsv41-lanes, tf-dsv41-rank-cache, ...)
  #   nix build .#tools          # the prepare tools: hf, python3 with torch (CPU), safetensors, tokenizers, Pillow
  #   nix develop                # a shell with both
  # The pins live in flake.lock: change a rev below and run `nix flake lock` (the patches are made for TensorFold's pin).
  description = "DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw on 2x DGX Spark with TensorFold's native Zig engine, without docker";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # The engine: BertholomusAI's TensorFold fork, branch deepseek-v41-zig, at the recipe's TF_REF (zig/patches apply to it)
    tensorfold = {
      url = "git+https://github.com/bertholomus/TensorFold?ref=deepseek-v41-zig&rev=629c524a0ea57512a5342abb498d1b7c25353343&shallow=1";
      flake = false;
    };
    # The inputs' tools: the same fork's branch deepseek-v41-zig-tp4 at INPUTS_REF (only tf-dsv41-rank-cache and the scripts)
    tensorfold-inputs = {
      url = "git+https://github.com/bertholomus/TensorFold?ref=deepseek-v41-zig-tp4&rev=59402086381a42c74d53053443224eec2790b4c5&shallow=1";
      flake = false;
    };
    # The family's kernel kit at KIT_REF (prepare copies it next to the checkpoint and adds the RoPE tables and image bias)
    kit = {
      url = "git+https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10?ref=main&rev=abe07c7d509a0199cc7fec15263be2aa52ac652f&shallow=1";
      flake = false;
    };
    # The checkpoint, Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw, as its git repository WITHOUT the LFS
    # objects: config, tokenizer, index and each shard's LFS pointer (sha256 and size, ~7 MB). bin/prepare downloads the
    # 196 GiB of weights at this revision with `hf` and checks every shard against these pointers. Not shallow: the Hub
    # serves only branch tips by hash, and a full fetch of main still finds this commit after later card edits.
    model = {
      url = "git+https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw?ref=main&rev=45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, tensorfold, tensorfold-inputs, kit, model }:
    let
      systems = [ "aarch64-linux" "x86_64-linux" ];   # the Sparks are aarch64; x86_64 builds the same for checking
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system);
      pkgsFor = system: import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;              # CUDA's redistributables (nvcc, cudart, cuBLAS) are unfree
          cudaCapabilities = [ "12.1" ];   # the GB10's sm_121 (NCCL's own kernels too)
          cudaForwardCompat = false;
        };
      };
    in
    {
      packages = forAll (system:
        let
          pkgs = pkgsFor system;
          cuda = pkgs.cudaPackages_13;
          zig = pkgs.callPackage ./nix/zig.nix { };
          # The NCCL the recipe measured with (NVIDIA's PyTorch container 26.07 ships 2.30.7; nixpkgs has 2.28)
          nccl = cuda.nccl.overrideAttrs (_: {
            version = "2.30.7-1";
            src = pkgs.fetchFromGitHub {
              owner = "NVIDIA";
              repo = "nccl";
              rev = "v2.30.7-1";
              hash = "sha256-fdiQZweX0jYfGroP0bL5Sfv3+DkCzVBZZLEbPv8aqq8=";
            };
          });
          # what the engine starts for pictures other than PNG (`python3 -I` with Pillow; the served lane's is 12.3.0)
          enginePython = pkgs.python3.withPackages (ps: [ ps.pillow ]);
          engine = pkgs.callPackage ./nix/tensorfold.nix {
            inherit zig cuda nccl enginePython;
            src = tensorfold;
            inputsSrc = tensorfold-inputs;
            patches = ./zig/patches;
            pins = {
              tf = tensorfold.rev;
              inputs = tensorfold-inputs.rev;
              kit = kit.rev;
              model = model.rev;
            };
          };
          tools = pkgs.callPackage ./nix/tools.nix { };
          modelMeta = pkgs.runCommand "dsv41-1017-exl3-meta-${builtins.substring 0 8 model.rev}" { } ''
            mkdir -p $out
            cp -r ${model}/. $out/
          '';
          kitSrc = pkgs.runCommand "dsv41-kit-${builtins.substring 0 8 kit.rev}" { } ''
            mkdir -p $out
            cp -r ${kit}/. $out/
          '';
        in
        {
          default = engine;
          tensorfold = engine;
          inherit zig nccl tools;
          # bin/prepare copies the worker's files with it (the same store path on both Sparks: --rsync-path)
          inherit (pkgs) rsync;
          model-meta = modelMeta;
          kit = kitSrc;
        });

      # services.dsv41-tensorfold: one rank as dsv41-tensorfold.service (nix/module.nix; not started at boot)
      nixosModules.default = import ./nix/module.nix {
        inherit self;
        pins = { model = model.rev; kit = kit.rev; };
      };

      devShells = forAll (system:
        let p = self.packages.${system}; pkgs = pkgsFor system; in {
          default = pkgs.mkShell { packages = [ p.tensorfold p.tools p.zig ]; };
        });
    };
}
