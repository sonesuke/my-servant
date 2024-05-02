set -eu

# Prepare onnxruntime and open_jtalk for voicevox_core


binary=download-osx-arm64
curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/latest/download/${binary} -o download
chmod +x download
./download

rm download
sudo cp voicevox_core/libonnxruntime*.dylib /usr/local/lib

huggingface-cli download lightblue/suzume-llama-3-8B-multilingual-gguf ggml-model-Q4_K_M.gguf --local-dir ./models/suzume-llama-3-8b-mul --local-dir-use-symlinks False

ollama create suzume-mul -f models/suzume-llama-3-8b-mul/Modelfile