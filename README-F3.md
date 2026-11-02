# F3: Strata on 4x Intel Arc Pro B70 at 512K context

F3 is a small, tested build of [Strata](https://github.com/Niko1221/Strata) for a workstation with **four Intel Arc Pro B70 (32 GB) cards**. It runs **Qwen3.8-Flash-Next UD-Q4_K_XL** across all four cards with a layer split, a **524,288-token (512K) context** (YaRN x2), and MTP speculative decoding.

It's shared as-is to help other people running multiple B70s. It's not an official Strata release.

## What's in this branch

The base is **hotschmoe/Strata `1c360526`** (`1c360526bce659f45cec60b2e36c46b7770c5bf1`). On top of it are two commits cherry-picked from **[Strata PR #3248](https://github.com/Niko1221/Strata/pull/3248) by maxfridbe** (still open upstream as of 2026-11-02), with authorship kept:

| Commit | Upstream commit | What it does |
|---|---|---|
| `f43afbf7` | `d95a9c4b` | sycl: host-only copies go through the CPU, because the Arc's copy engine hangs on pinned-host <-> pageable copies (engine reset/wedge on B70) |
| `87a75c05` | `8cc654f2` | sycl: QSA block scores as oneMKL GEMM tiles from 8K cells (faster long prompts) |

Applying `8cc654f2` onto `1c360526` conflicts in `sycl/src/prefill/prefill.cpp` (a block of declarations) and `docs/INTEL.md`; both were resolved by taking the incoming side, with no other edits. 5 files changed, +182/-4. The resulting `sycl/` tree is `608eb3b1743964ab4de8d7c47e70386d6c5e7b79`, and an independent re-application of the two public commits produced the same tree.

All of the real engineering credit goes to **Niko1221 and the Strata contributors**, **hotschmoe** (the fork this is based on) and **maxfridbe** (PR #3248). What we added is the testing on 4x B70 and the settings below.

**Optional extra** (`f3/optional/serve-fixes-2141-3441-3075.diff`): our running build also patches the Python server (`serve/`) with two upstream fixes: #2141, honour `thinking_budget_tokens` (by xrip, from PR #2140), and #3441/#3075, tool-call parsing (by Niko1221). They don't touch the engine and aren't in this branch's commits. Apply them yourself if you want them.

## Hardware and software we used

| Item | Value |
|---|---|
| CPU | AMD Threadripper PRO 5965WX |
| GPUs | 4x Intel Arc Pro B70 32 GB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-38-generic |
| Intel compute runtime | 26.31.39395.16-1~26.04~ppa1 |
| Level Zero loader (libze1) | 1.32.0-1~26.04~ppa1 |
| Compiler | Intel oneAPI DPC++/C++ 2026.1.1 (icx/icpx), oneAPI 2026.1 |
| Build | Release, Ninja, AOT for `bmg-g31` (B70) |
| Model | [unsloth Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth), UD-Q4_K_XL (4 shards) |

These versions come from our build notes. Other versions may work, but we haven't tested them.

## Build

These are the exact commands our builds used (native, no Docker):

```bash
git clone https://github.com/Mouzon1775/Strata.git
cd Strata
git checkout f3-4xb70
source /opt/intel/oneapi/setvars.sh
cmake -S sycl -B build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DSTRATA_SYCL_AOT=bmg-g31
cmake --build build --target strata -j 24
# binary: build/strata
```

The fork's `sycl/CMakeLists.txt` pins its ggml dependency (`3cf03257`), so the configure step downloads it. The Docker route in `docs/INTEL.md` (`strata-sycl-dev` image, `AOT=bmg-g31`) should also work, but we didn't use it.

You make the model pack, MTP files and expert profile the normal Strata Intel way (`sycl/setup_intel.py`, see `docs/INTEL.md`).

## Launch

`f3/launch/f3-512k.example.json` is our server config with every private path replaced by a placeholder. It has the full environment (43 settings) and every engine flag. Start it with:

```bash
python3 sycl/serve/server_intel.py --engine strata --config f3/launch/f3-512k.example.json \
    --host 127.0.0.1 --port 8080 --tokenizer /path/to/pack/tokenizer
```

`server_intel.py` adds `--serve --layer-split auto`. `f3/launch/f3-512k-engine.example.sh` shows the same engine command run directly. The key engine flags are:

```
--spec 6 --mtp-max-t 6 --spec-min-p 0.5 --mtp /path/to/mtp/rt
--max-context 524288 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
--kv int8 --stream-experts --expert-cache auto --prefill 8192 --no-prefill-borrow
--vram-reserve-mib 1024 --conversation-cache-mib 32768 --conversation-cache-slots 4
--layer-split auto   (gpu 0,1,2,3)
```

## Context length: 512K or 262K

The model was trained to 262,144 tokens; 512K needs YaRN (`--rope-scaling yarn --rope-scale 2`), which applies to every prompt, short ones too. Recall at 500K tokens was perfect in our tests, but on our own TMDL (Power BI model code) editing tests 512K scored slightly worse than the native 262K, so **our everyday setting is 262K** (drop the three YaRN flags and set `--max-context 262144`) and we switch to 512K only for sessions that need it.

## Measured results

These are only numbers we measured ourselves. Your results will vary.

**MTP sweep, 2026-11-01** (all 4 B70s, 512K context, settings above, `--spec 6 --mtp-max-t 6 --spec-min-p 0.5`). The engine was built from the same SYCL source tree as this branch (tree `608eb3b1`), with the optional server fixes applied.

| Metric | Result |
|---|---|
| Generation speed, coding tasks (mean of 6 tasks at 32K / 100K / 200K context) | **95.1 tok/s** (per task 90.1-99.8) |
| Generation speed, short prompts | 111.5 tok/s |
| Prompt speed, 208,616-token prompt (mean of 3) | 3,953 tok/s |
| Time to first token at that length | 52.9 s |
| MTP draft acceptance, coding | 76.6% |
| Host RAM peak | 29.4 GiB |
| GPU busy, average across all 4 cards | 44.5% |
| GPU faults (xe fault/reset/wedge) | none across about 4.75 hours |

In the same sweep, depth 7 / min-p 0.7 was about 2.4% faster on generation (97.5 tok/s) with byte-identical output. We kept 6/0.5.

**PR #3248 effect, 2026-10-20** (same hardware, compared with plain hotschmoe 1c360526): about 3.3 s faster to first token on a 100K prompt (-8.8%), about 0.5 s faster at 32K. Decode output was byte-identical to the unpatched build on every comparable prompt.

Not measured, so not claimed: other models, other quants, fewer than 4 cards, and other B-series cards.

## Licence

MIT, unchanged from Strata (`LICENSE`, "Copyright (c) 2026 Niko1221 and the Strata contributors").
