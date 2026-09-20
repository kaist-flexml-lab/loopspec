# LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers

SangLyul Cho*, Langqing Cui*, Sehoon Kim, Dongsu Han, Insu Han<br>
Seoul National University · KAIST · *Equal contribution

[Paper](https://arxiv.org/pdf/2609.17184) | [Project Page](https://langq1225.github.io/loopspec/) | [Getting Started](#getting-started) | [Reproducing Benchmarks](#reproducing-benchmarks) | [Inference](#inference)

Accelerate Ouro and Raven inference using the model's own intermediate predictions. LoopSpec drafts upcoming tokens while verifying earlier ones, reusing work across recurrent steps.

- **Use existing checkpoints.** No training or separate draft model required.
- **Keep the target model's behavior.** Supports lossless greedy decoding and sampling.
- **Run through SGLang.** Includes an inference server, a streaming demo, and benchmark scripts.

LoopSpec achieves **up to 6.83× decoding speedup**. See the [paper](https://arxiv.org/pdf/2609.17184) for results and the method.

## Getting Started

Requires **Linux x86-64, Python 3.11, `uv`, and an NVIDIA GPU** with a compatible CUDA environment. Dependencies are pinned in [pyproject.toml](pyproject.toml) and `uv.lock`.

```bash
git clone https://github.com/kaist-flexml-lab/loopspec.git
cd loopspec
uv sync --locked --no-dev
```

Download the model checkpoints used in the benchmarks:

```bash
uv run --locked --no-dev download.py
```

## Reproducing Benchmarks

The benchmark runner compares baseline and LoopSpec through `lm-evaluation-harness`. It uses **Docker Compose and the NVIDIA Container Toolkit** to build and launch the inference and evaluation containers. The [Dockerfile](Dockerfile) provides the CUDA 13.3 environment.

Choose an experiment, download its models, and update its `gpus` list for your machine (for example, `gpus: [0]`). Remove models or tasks from the YAML to run a smaller comparison.

| Experiment | Tasks |
| --- | --- |
| [Math](experiments/110-speed-math.yaml) | GSM8K, MATH-500 |
| [Thinking models](experiments/111-speed-math-chat.yaml) | GSM8K, MATH-500, AIME 2024/2025; greedy and sampling |
| [Code generation](experiments/112-speed-coding.yaml) | HumanEval+, MBPP+ |
| [General reasoning](experiments/113-speed-general.yaml) | BBH |

Create the shared cache volume once, then launch an experiment:

```bash
docker volume create pipelined-model-cache
uv run --locked --no-dev python benchmark.py run experiments/110-speed-math.yaml
```

Each job saves evaluation outputs, generation traces, and server logs under `results/<model>/<task>/<configuration>/`. Summarize completed runs with:

```bash
uv run --locked --no-dev python benchmark.py report results
```

For comparable latency measurements, keep `max_running_requests: 1` and `num_concurrent: 1`. Baseline accuracy runs are provided separately for [base models](experiments/100-accuracy-base.yaml) and [thinking models](experiments/101-accuracy-thinking.yaml). Prompting and token budgets are configured in [experiments/tasks.yaml](experiments/tasks.yaml).

## Inference

Start a LoopSpec server on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-dev python -m sglang_recurrent.server \
  --model models/Ouro-1.4B \
  --host 127.0.0.1 \
  --port 30000 \
  --loopspec --step-size 1 --second-step 2
```

Once the server is ready, generate text from another terminal:

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 128}
  }'
```

Set `"temperature": 1.0` and `"top_p": 0.7` in `sampling_params` to use sampling.

### Compare decoding modes

Use the same launch command with the following decoding flags:

| Mode | Flags | Experiment config key |
| --- | --- | --- |
| Autoregressive baseline | Omit all LoopSpec flags | `baseline` |
| LoopSpec, one proposal (Ouro) | `--loopspec --step-size 1` | `"1"` |
| LoopSpec, two proposals (Ouro) | `--loopspec --step-size 1 --second-step 2` | `"1:2"` |
| LoopSpec, two proposals (Raven) | `--loopspec --step-size 2 --second-step 4` | `"2:4"` |

You should also change `--model` to a downloaded Raven checkpoint. `--step-size K` sets the first proposal depth; `--second-step X` places the second proposal at depth `K × X`. Thus, `"2:4"` means proposals at depths 2 and 8. `K` must divide the model's total recurrent depth `R`, and a second proposal requires `1 < X < R/K`.

### Interactive demo

The terminal demo streams generated text and reports decoding speed. In [demo/config.yaml](demo/config.yaml), set `active.model` to a downloaded model, `active.config` to a config key above, and `server.gpu` to your GPU index. Then run:

```bash
bash demo/run.sh
```

Enter prompts interactively and use `/quit` to stop. See [demo/README.md](demo/README.md) for one-shot usage and other options.

## Supported Models

Use the original checkpoints below. Local model directories should use the checkpoint name, for example `models/Ouro-1.4B`.

| Model | Checkpoint |
| --- | --- |
| Ouro-1.4B | `ByteDance/Ouro-1.4B` |
| Ouro-2.6B | `ByteDance/Ouro-2.6B` |
| Ouro-1.4B-Thinking | `ByteDance/Ouro-1.4B-Thinking` |
| Ouro-2.6B-Thinking | `ByteDance/Ouro-2.6B-Thinking` |
| Raven-Llama-3.2 | `smcleish/Recurrent-Llama-3.2-train-recurrence-32` |
| Raven-OLMo-2-0425 | `smcleish/Recurrent-OLMo-2-0425-train-recurrence-32` |
| Raven-TinyLlama-3T | `smcleish/Recurrent-TinyLlama-3T-train-recurrence-32` |

To download all seven models:

```bash
uv run --locked --no-dev python download.py
```

## Citation

If you use LoopSpec in your research, please cite:

```bibtex
@misc{cho2026loopspec,
      title={LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers}, 
      author={SangLyul Cho and Langqing Cui and Sehoon Kim and Dongsu Han and Insu Han},
      year={2026},
      eprint={2609.17184},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.17184}, 
}
```
