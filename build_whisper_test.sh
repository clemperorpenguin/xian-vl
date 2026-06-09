#!/bin/bash
set -e
git clone -b v1.8.4 https://github.com/ggerganov/whisper.cpp /tmp/whisper_build
cd /tmp/whisper_build
cmake -B build -DWHISPER_VULKAN=1 -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j --target server
