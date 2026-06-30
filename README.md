# llm-tracer

A lightweight diagnostic tool that hooks non-invasively into a local transformer model (via llama.cpp) and captures real-time intermediate states — layer latencies, activation stats, attention matrices — displayed in an interactive terminal UI.

Built in C++ for GDSC IIT Roorkee Open Projects Summer '26.

## Team Info
- Team  - LLM_11
- Teammates - Krishan And Indra
---

## What it does

- **Non-invasive hook** into llama.cpp's eval callback — no source modification
- **Captures per-tensor:** shape, dtype, latency, sparsity, mean, max
- **Detects anomalies:** outlier activations, CUDA fallbacks, latency spikes
- **Interactive TUI** with 5 panels: model topology, live event stream, attention matrix, runtime metrics, anomaly ledger

### Why llm-tracer? (Key Differentiators)
While there are many tools for logging or profiling ML models, `llm-tracer` stands out by focusing on **real-time, local observability without overhead**:
- **Zero Python Overhead:** Written entirely in C++, avoiding the sluggishness of Python-based visualization tools.
- **True Non-Invasiveness:** Does not require maintaining a custom fork of `llama.cpp`. You drop in the submodule, and it hooks via standard C APIs.
- **Immediate TUI Feedback:** Instead of parsing static JSON logs after a run, you watch the model's internal state (attention sparsity, latency spikes) evolve live in your terminal.

---

## Prerequisites

| Tool | Min version | Notes |
|---|---|---|
| CMake | 3.16 | `brew install cmake` on macOS |
| C++ compiler | C++17 | Xcode CLT on macOS, `gcc` on Linux |
| Git | any | for submodule init |

No Python. No pip. No conda. Pure C++.

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd llm-tracer

# 2. Pull llama.cpp (don't clone separately — use the submodule)
git submodule update --init --recursive

# 3. Download the dev model (~670 MB)
bash scripts/download_model.sh

# 4. Build everything
bash scripts/build.sh
```

### Troubleshooting
- **CMake errors:** Ensure you have CMake 3.16+ installed (`cmake --version`). On macOS, you may need to install the Xcode Command Line Tools (`xcode-select --install`).
- **Missing headers:** If `llama.h` or similar cannot be found, ensure you ran step 2 properly (`git submodule update --init --recursive`).

---

## Run

```bash
# Baseline inference (Phase 0 — confirms the stack works)
./build/llm_tracer --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "The sky is"

# Hook dump test (Phase 0 — confirms the hook mechanism captures tensor events)
./build/test_hook_dump --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Full TUI execution (Target end-state)
# ./build/llm_tracer_tui --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## Project structure

```
llm-tracer/
├── extern/llama.cpp/        # llama.cpp as git submodule (do not modify)
├── include/                 # Header files (.hpp)
│   ├── core/                # LayerEvent, RingBuffer, AttentionStore
│   ├── hook/                # HookManager, AnomalyDetector
│   ├── topo/                # TopologyParser
│   └── tui/panels/          # One header per TUI panel
├── src/                     # Implementation files (.cpp)
├── tests/                   # One test binary per phase
├── models/                  # GGUF files go here (gitignored)
├── scripts/                 # build.sh, download_model.sh
└── CMakeLists.txt
```

---

## Build phases

| Phase | What gets built | Status |
|---|---|---|
| 0 | Scaffold + baseline inference | ✓ done |
| 1 | Core data structures (LayerEvent, RingBuffer, AttentionStore) | — |
| 2 | Hook mechanism (HookManager, AnomalyDetector) | — |
| 3 | Model topology parser | — |
| 4 | TUI skeleton (ftxui, 5 panels, keyboard) | — |
| 5 | Panel implementations | — |
| 6 | Full integration + threading | — |
| 7 | Polish + edge cases | — |
| 8 | Demo prep + submission | — |

---

## Notes on the hook mechanism

llama.cpp exposes `ggml_backend_sched_set_eval_callback()` — a published C API that fires before and after every tensor computation. This is how the tool hooks in without touching the model's source code.

The callback receives a `ggml_tensor*` with name, shape, dtype, and a data pointer. From this, all metrics are derived.

---

## Assumptions & Limitations

- **Backend Support:** Currently assumes CPU execution for extracting tensor data pointers directly. CUDA/Metal support requires additional synchronization logic (e.g., copying device data to host) which is planned for later phases.
- **Model Format:** Assumes models are in GGUF format and compatible with the currently linked `llama.cpp` submodule.
- **Topology:** Assumes standard transformer architectures. Highly exotic architectures might not render their topology perfectly in the TUI, though raw tensor events will still be captured.

---

## Verification Strategy

For reviewers and evaluators testing this project, you can verify its functionality as follows:

1. **Verify the Hook (Phase 0/2):** Run `./build/test_hook_dump --model <model>`. You should see a stream of console outputs detailing tensor shapes and latencies, proving the non-invasive hook works during inference.
2. **Verify Anomaly Detection (Future Phase):** Once the TUI is implemented, you will be able to pass an `--anomaly-threshold` flag (e.g., `--anomaly-threshold 0.001`). Setting this to an artificially low value will intentionally flag normal activations as anomalies, demonstrating the real-time anomaly ledger in the UI.
3. **Verify Performance:** Run inference with and without `llm-tracer` attached. The difference in tokens-per-second (TPS) demonstrates the low-overhead nature of our C++ implementation.
