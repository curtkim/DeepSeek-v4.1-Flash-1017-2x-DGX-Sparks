# What bin/prepare runs on the head besides the engine, in place of the container's Python: the Hugging Face CLI (`hf`,
# with hf-xet for the Hub's storage), and a Python with torch on CPU (make_rope.py, the ABLIT overlay), safetensors,
# tokenizers (make_token_map.py) and Pillow. No CUDA: nothing here touches the GPU.
{ python3, symlinkJoin }:
let
  py = python3.withPackages (ps: with ps; [
    huggingface-hub
    hf-xet
    torch
    safetensors
    tokenizers
    pillow
    numpy
  ]);
in
symlinkJoin {
  name = "dsv41-prepare-tools";
  paths = [ py ];
  meta.description = "hf and a CPU Python (torch, safetensors, tokenizers, Pillow) for bin/prepare";
}
