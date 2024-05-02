set -eu

# Prepare onnxruntime and open_jtalk for voicevox_core


binary=download-osx-arm64
curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/latest/download/${binary} -o download
chmod +x download
./download

rm download
sudo cp voicevox_core/libonnxruntime*.dylib /usr/local/lib

