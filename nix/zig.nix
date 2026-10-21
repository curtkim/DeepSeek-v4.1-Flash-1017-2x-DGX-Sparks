# Zig 0.17.0, the version TensorFold's build.zig.zon asks for (nixpkgs stops at 0.16): ziglang.org's static release, as
# zig/Dockerfile installs it (same sha256 for aarch64).
{ lib, stdenvNoCC, fetchurl }:
let
  version = "0.17.0";
  release = {
    aarch64-linux = { arch = "aarch64"; sha256 = "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8"; };
    x86_64-linux = { arch = "x86_64"; sha256 = "1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026"; };
  }.${stdenvNoCC.hostPlatform.system} or (throw "zig ${version}: no release for ${stdenvNoCC.hostPlatform.system}");
in
stdenvNoCC.mkDerivation {
  pname = "zig";
  inherit version;
  src = fetchurl {
    url = "https://ziglang.org/download/${version}/zig-${release.arch}-linux-${version}.tar.xz";
    inherit (release) sha256;
  };
  dontConfigure = true;
  dontBuild = true;
  dontFixup = true;   # a static binary that finds lib/ next to itself
  # zig finds its lib/ beside the executable's real path (/proc/self/exe), so the release stays together
  installPhase = ''
    mkdir -p $out/libexec/zig $out/bin
    cp -r . $out/libexec/zig/
    ln -s ../libexec/zig/zig $out/bin/zig
  '';
  meta = {
    description = "Zig ${version} (official static release)";
    license = lib.licenses.mit;
    platforms = [ "aarch64-linux" "x86_64-linux" ];
    mainProgram = "zig";
  };
}
