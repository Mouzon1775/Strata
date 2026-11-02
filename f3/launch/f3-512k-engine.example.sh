#!/usr/bin/env bash
# F3 engine started directly (what server_intel.py runs). Replace every /path/to/... first.
# Set the same environment as the "env" block in f3-512k.example.json before running.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
cd /path/to/Strata
exec ./build/strata --serve \
  --pack /path/to/pack \
  --native /path/to/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --ple-gguf /path/to/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  --expert-profile /path/to/Strata/data/expert-profile-sycl.bin --expert-cache auto \
  --prefill 8192 --no-prefill-borrow \
  --spec 6 --spec-min-p 0.5 --mtp /path/to/mtp/rt --mtp-max-t 6 \
  --suffix-draft 0 --lookup-chain 0 \
  --max-context 524288 --kv int8 --stream-experts --vram-reserve-mib 1024 \
  --conversation-cache-mib 32768 --conversation-cache-slots 4 \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 \
  --layer-split auto
