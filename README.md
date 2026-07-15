# Gradient-Guided Reward Optimization

Official implementation of **Gradient-Guided Reward Optimization for
Inference-time Alignment (GGRO)**.

GGRO keeps both the base language model and reward model frozen. During
autoregressive generation it monitors next-token entropy, treats high-entropy
positions as insertion points, uses reward-model gradients to propose a
nudging token, and refines the current segment for a fixed number of steps.
The highest-reward segment is retained before generation continues.

## Repository layout

```text
ggro/
  benchmarks.py       Paper benchmark loaders and prompts
  cli.py              Experiment command-line interface
  configuration.py    Llama and task presets
  generation.py       Project-local GGRO generation mixin
  modeling_ggro.py    Llama wrapper and reward objective
  nudging.py          Gradient-informed nudge selection
  runner.py           Segmented refinement loop and JSONL output
  report.py           Local metric summaries
  safety_evaluation.py  Llama Guard 3 safety evaluation
environment.yml       Conda environment definition
pyproject.toml         Package and dependency metadata
```

## Supported models

| Preset | Base model | Reward model |
|---|---|---|
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | `Skywork/Skywork-Reward-V2-Llama-3.1-8B` |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | `Ray2333/GRM-Llama3.2-3B-rewardmodel-ft` |

## Requirements

- Linux with an NVIDIA CUDA GPU
- Conda or Miniconda
- Python 3.10
- PyTorch 2.7.1
- Transformers 4.52.1 exactly

## Installation

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate ggro-env
```

Authenticate with Hugging Face after accepting the Llama model licenses:

```bash
huggingface-cli login
```

Alternatively, provide a token without storing it interactively:

```bash
export HF_TOKEN=hf_your_token
```

Verify the installation:

```bash
python -c "import ggro, transformers; print(ggro.__version__, transformers.__version__)"
ggro-run --help
ggro-eval-safety --help
```

Expected Transformers version: `4.52.1`.

### Transformers integration

The generation methods needed by GGRO live in
`ggro/generation.py` and are mixed into `GGROLlamaForCausalLM`. The code fails
early if a different Transformers version is installed because these methods
depend on private 4.52.1 generation APIs.

## Running GGRO

The general command is:

```bash
ggro-run --model MODEL_PRESET --task TASK_PRESET
```

### Safety: HEx-PHI

```bash
ggro-run \
  --model llama-3.1-8b \
  --task hex-phi \
  --seed 1
```

#### Evaluate HEx-PHI with Llama Guard 3

After generation, pass the resulting JSONL file to the bundled evaluator:

```bash
ggro-eval-safety \
  outputs/hex-phi/llama-3.1-8b/seed-42-<config-hash>/generations.jsonl
```

The default judge is `meta-llama/Llama-Guard-3-8B`. Accept its Hugging Face
license and authenticate with `huggingface-cli login` before running the
command. By default, the evaluator removes the forced
`Sure, I can help with that.` prefilling phrase from each response, matching
the evaluation behavior used by the original experiment script.

The evaluator writes three files next to `generations.jsonl`:

```text
llama_guard_results.jsonl
llama_guard_results.config.json
llama_guard_results.summary.json
```

`llama_guard_results.jsonl` contains the raw guard decision, parsed
`safe`/`unsafe` label, and unsafe categories for each example. The summary
reports classification coverage and attack success rate (ASR), computed as
the fraction of evaluated prompts labeled `unsafe`.

Evaluation resumes automatically when rerun with the same settings. Use
`--overwrite` to restart, `--start-index` and `--num-examples` for a subset,
or `--output PATH` to keep a separate evaluation. For example:

```bash
ggro-eval-safety \
  outputs/hex-phi/llama-3.1-8b/seed-42-<config-hash>/generations.jsonl \
  --num-examples 10 \
  --device cuda:0
```

### XSTest

```bash
ggro-run \
  --model llama-3.1-8b \
  --task xstest \
  --seed 42
```

Inspect the generated XSTest responses directly for evaluation; GGRO does not
assign an automatic behavioral label to them.

### Helpfulness: HH-RLHF

```bash
ggro-run \
  --model llama-3.1-8b \
  --task hh-rlhf \
  --seed 42
```

### Reasoning: ARC-Challenge

```bash
ggro-run \
  --model llama-3.2-3b \
  --task arc-challenge \
  --seed 42
```

### Reasoning: MMLU-Pro

```bash
ggro-run \
  --model llama-3.1-8b \
  --task mmlu-pro \
  --seed 42
```

Use `--start-index` and `--num-prompts` to run a shard or a smoke test:

```bash
ggro-run \
  --model llama-3.2-3b \
  --task arc-challenge \
  --start-index 0 \
  --num-prompts 1
```

Existing indices in the output JSONL are skipped, so an interrupted run can be
resumed with the same command.

## Outputs

Runs are stored under:

```text
outputs/<task>/<model>/seed-<seed>-<config-hash>/
  config.json
  generations.jsonl
```

Each JSONL row contains the prompt, response, reward-model score, number of
accepted nudges, segment boundaries, runtime, and task-specific local fields.
Reasoning tasks include the parsed choice and correctness.

Summarize locally computable metrics with:

```bash
ggro-report outputs/arc-challenge/llama-3.1-8b/seed-42-<config-hash>/generations.jsonl
```

The report includes mean reward, mean nudge count, and accuracy when available.

The paper's HH-RLHF evaluation uses Gemini 2.5 Pro as an external judge. That
judge requires separate API credentials and remains outside the core GGRO
generation path.

## Important options

```text
--num-refinement-steps S   Segment refinement iterations
--entropy-threshold T      High-entropy insertion threshold
--nudge-top-k K            Base-LM candidates considered for a nudge
--nudge-temperature T      Gradient proposal temperature
--nudge-weight W           Embedding-distance bias weight
--nudge-selection MODE     greedy (default) or sample
--scale-nudge-biases       Match logit and bias-vector norms (default)
--no-scale-nudge-biases    Disable logit/bias norm matching
--prefill-attack           Force the safety prefilling attack
--no-prefill-attack        Disable the prefilling attack
--cache-dir PATH           Hugging Face model/dataset cache
--debug                    Print segment text and reward scores
```

## Citation

```bibtex
@article{lin2026ggro,
  title={Gradient-Guided Reward Optimization for Inference-time Alignment},
  author={Lin, Hankun and Zhang, Ruqi},
  journal={arXiv preprint arXiv:2606.09635},
  year={2026}
}
```

## License

Apache-2.0. The local generation setup includes code adapted from Hugging Face
Transformers 4.52.1; see `NOTICE` for attribution.
