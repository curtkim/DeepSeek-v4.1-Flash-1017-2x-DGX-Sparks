# TensorFold's native engine for the DeepSeek-V4.1-Flash family, as zig/Dockerfile builds it, without the container:
#   - the serving engine: TensorFold at the deepseek-v41-zig pin plus zig/patches (git apply, filename order), built with
#     `zig build -Dnvcc=... -Dsm=121 -Doptimize=ReleaseFast install native` (the kernels' fatbins embedded);
#   - the inputs' tools of the deepseek-v41-zig-tp4 branch: tf-dsv41-rank-cache (a rank's weight file from the checkpoint,
#     no GPU) and make_token_map.py, make_rope.py, make_bias_vl.py.
# The binaries dlopen libcuda.so.1, libcublas(Lt).so.13, libnccl.so.2 and libibverbs.so.1 and start `python3` for
# pictures other than PNG. Each bin/ entry is a wrapper that puts this package's CUDA 13 libraries, NCCL and rdma-core in
# LD_LIBRARY_PATH (NCCL dlopens libibverbs from its own library, so a RUNPATH on the engine would not reach it), the host's
# NVIDIA driver after them (libcuda.so.1 and its libnvidia-* only, linked into a private folder, never the host's whole
# library folder), and the Python with Pillow first in PATH.
{ lib
, cuda
, nccl
, zig
, enginePython
, git
, patchelf
, rdma-core
, runtimeShell
, src
, inputsSrc
, patches
, pins
, sm ? "121"
}:
let
  stdenv = cuda.backendStdenv;   # the gcc nvcc accepts as its host compiler
  # (CUDA 13 moved CCCL's headers to include/cccl: both are given, a missing one is ignored)
  includes = lib.concatMap (p: [ "${lib.getInclude p}/include" "${lib.getInclude p}/include/cccl" ])
    [ cuda.cuda_cudart (cuda.cccl or cuda.cuda_cccl) cuda.cuda_crt ];
  runtimeLibs = [ (lib.getLib cuda.libcublas) (lib.getLib nccl) (lib.getLib rdma-core) ];
  # every binary of the engine and the inputs' tools that this recipe runs or that helps to look into a rank
  programs = [
    "native/bin/tensorfold-native"
    "bin/tf-dsv41-lanes"
    "bin/tf-dsv41-rank-cache"
    "bin/tf-dsv41-check"
    "bin/tf-dsv41-load"
    "bin/tf-dsv41-comm"
    "bin/tf-dsv41-rdma-bench"
    "bin/tf-dsv41-engram-check"
    "bin/tf-dsv41-picture"
    "bin/tensorfold"
  ];
in
stdenv.mkDerivation {
  pname = "tensorfold-dsv41";
  version = "1.0.2-unstable-${builtins.substring 0 8 pins.tf}";
  inherit src;

  nativeBuildInputs = [ zig git patchelf ];
  dontConfigure = true;
  # zig's ReleaseFast output: no stripping by the generic fixup (it strips the embedded fatbins' symbols it does not know)
  dontStrip = true;

  postPatch = ''
    for p in ${patches}/*.patch; do
      [ -e "$p" ] || continue
      echo "applying $(basename "$p")"
      git apply --check "$p"
      git apply "$p"
    done
  '';

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    export ZIG_GLOBAL_CACHE_DIR=$TMPDIR/zig-global ZIG_LOCAL_CACHE_DIR=$TMPDIR/zig-local
    # nvcc with the CUDA headers of this package set (the kernels include cuda_runtime.h, cuda_bf16.h, cooperative_groups.h);
    # its host compiler is the gcc on PATH (cuda.backendStdenv's)
    cat > $TMPDIR/nvcc <<EOF
    #!${runtimeShell}
    exec ${cuda.cuda_nvcc}/bin/nvcc ${lib.concatMapStringsSep " " (i: "-I${i}") includes} "\$@"
    EOF
    chmod +x $TMPDIR/nvcc

    zig build -Dnvcc=$TMPDIR/nvcc -Dsm=${sm} -Doptimize=ReleaseFast --prefix $TMPDIR/engine -j$NIX_BUILD_CORES install native
    test -x $TMPDIR/engine/native/bin/tensorfold-native
    test -x $TMPDIR/engine/bin/tf-dsv41-lanes

    cp -r ${inputsSrc} $TMPDIR/inputs-src
    chmod -R u+w $TMPDIR/inputs-src
    (cd $TMPDIR/inputs-src && zig build -Dnvcc=$TMPDIR/nvcc -Dsm=${sm} -Doptimize=ReleaseFast --prefix $TMPDIR/inputs -j$NIX_BUILD_CORES install)
    test -x $TMPDIR/inputs/bin/tf-dsv41-rank-cache
    cp $TMPDIR/inputs/bin/tf-dsv41-rank-cache $TMPDIR/engine/bin/
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/libexec/tensorfold $out/share/tensorfold/tools/inputs $out/share/doc/tensorfold
    cp -r $TMPDIR/engine/. $out/libexec/tensorfold/
    cp $TMPDIR/inputs-src/tools/dsv41-zig/inputs/make_token_map.py \
       $TMPDIR/inputs-src/tools/dsv41-zig/kit/make_rope.py \
       $TMPDIR/inputs-src/tools/dsv41-zig/kit/make_bias_vl.py $out/share/tensorfold/tools/inputs/
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do
      if [ -e "$f" ]; then cp -a "$f" $out/share/doc/tensorfold/; fi
    done
    if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then cp tools/dsv41-zig/ATTRIBUTION.md $out/share/doc/tensorfold/DSV41-ATTRIBUTION.md; fi
    cp $TMPDIR/inputs-src/LICENSE $out/share/doc/tensorfold/LICENSE-inputs-branch

    # what the bin/ scripts read: the pins this package was built from
    cat > $out/share/tensorfold/pins.env <<EOF
    TF_REF=${pins.tf}
    INPUTS_REF=${pins.inputs}
    KIT_REF=${pins.kit}
    MODEL_REVISION=${pins.model}
    EOF

    # the environment every binary runs in (sourced by the wrappers; bin/start sources it too)
    cat > $out/libexec/tensorfold/env.sh <<'EOF'
    # The host's NVIDIA driver: TF_DRIVER_LIB_DIR, else NixOS's /run/opengl-driver/lib, else the folder that holds
    # libcuda.so.1 on a DGX OS / Ubuntu Spark. Only the driver's own libraries are linked into a private folder: the
    # host's library folder itself would put its libibverbs, NCCL or cuBLAS ahead of this package's.
    tf_driver_dir() {
      local d
      if [ -n "''${TF_DRIVER_LIB_DIR:-}" ]; then echo "$TF_DRIVER_LIB_DIR"; return 0; fi
      [ -e /run/opengl-driver/lib/libcuda.so.1 ] && { echo /run/opengl-driver/lib; return 0; }
      for d in /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/nvidia/lib64; do
        [ -e "$d/libcuda.so.1" ] || continue
        # under ~/.cache, not XDG_RUNTIME_DIR: a rank started over ssh outlives the session that made the folder
        local own="''${XDG_CACHE_HOME:-''${HOME:-/tmp}/.cache}/tensorfold-driver"
        mkdir -p "$own" || return 1
        for f in "$d"/libcuda.so* "$d"/libnvidia-*.so*; do
          if [ -e "$f" ]; then ln -sfn "$f" "$own/" 2>/dev/null || true; fi
        done
        echo "$own"; return 0
      done
      return 1
    }
    _tf_drv=$(tf_driver_dir) || echo "tensorfold: no libcuda.so.1 found (is the NVIDIA driver installed? TF_DRIVER_LIB_DIR names its folder)" >&2
    export LD_LIBRARY_PATH="@libs@''${_tf_drv:+:$_tf_drv}"
    export PATH="@python@/bin:$PATH"
    unset _tf_drv
    EOF
    substituteInPlace $out/libexec/tensorfold/env.sh \
      --replace-fail @libs@ ${lib.makeLibraryPath runtimeLibs} \
      --replace-fail @python@ ${enginePython}
    runHook postInstall
  '';

  # zig links against the target's default loader (/lib/ld-linux-*.so.1): give each binary this package set's glibc
  postFixup = ''
    for p in ${lib.concatStringsSep " " programs}; do
      real=$out/libexec/tensorfold/$p
      [ -x "$real" ] || { echo "missing $p" >&2; exit 1; }
      patchelf --set-interpreter "$(cat ${stdenv.cc}/nix-support/dynamic-linker)" "$real"
      cat > $out/bin/$(basename $p) <<EOF
    #!${runtimeShell}
    . $out/libexec/tensorfold/env.sh
    exec -a "\$0" $real "\$@"
    EOF
      chmod +x $out/bin/$(basename $p)
    done
  '';

  passthru = { inherit pins nccl enginePython; };

  meta = {
    description = "TensorFold's native Zig engine for DeepSeek-V4.1-Flash (BertholomusAI's deepseek-v41-zig) with this recipe's patches";
    homepage = "https://github.com/bertholomus/TensorFold";
    license = lib.licenses.asl20;
    platforms = [ "aarch64-linux" "x86_64-linux" ];
    mainProgram = "tensorfold-native";
  };
}
