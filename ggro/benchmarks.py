"""Dataset loading and prompt construction for the GGRO paper benchmarks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


@dataclass
class BenchmarkExample:
    prompt: str
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _multiple_choice_prompt(question: str, options: Iterable[str]) -> str:
    options_text = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"Human: {question}\n{options_text}\n"
        '\nThink step by step, then give your final answer in the form of "Answer: {Your Answer}."'
        "\n\nAssistant:"
    )


def _load_hex_phi() -> List[BenchmarkExample]:
    dataset = load_dataset("LLM-Tuning-Safety/HEx-PHI")
    splits = list(dataset.keys())
    max_examples = max(len(dataset[split]) for split in splits)
    examples = []
    for index in range(max_examples):
        for split in splits:
            if index < len(dataset[split]):
                content = dataset[split][index]["content"]
                examples.append(
                    BenchmarkExample(
                        prompt=f"Human: {content}\n\nAssistant:",
                        metadata={"category": split},
                    )
                )
    return examples


def _load_xstest() -> List[BenchmarkExample]:
    dataset = load_dataset("walledai/XSTest", split="test")
    return [
        BenchmarkExample(prompt=f"Human: {row['prompt']}\n\nAssistant:")
        for row in dataset
        if row["label"] == "safe"
    ]


def _load_hh_rlhf() -> List[BenchmarkExample]:
    dataset = load_dataset("Dahoas/full-hh-rlhf", split="test")
    return [BenchmarkExample(prompt=row["prompt"].removeprefix("\n\n")) for row in dataset]


def _load_arc_challenge() -> List[BenchmarkExample]:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    return [
        BenchmarkExample(
            prompt=_multiple_choice_prompt(row["question"], row["choices"]["text"]),
            answer=row["answerKey"],
        )
        for row in dataset
    ]


def _mmlu_example(row) -> BenchmarkExample:
    return BenchmarkExample(
        prompt=_multiple_choice_prompt(row["question"], row["options"]),
        answer=row["answer"],
        metadata={"category": row["category"]},
    )


def _load_mmlu_pro() -> List[BenchmarkExample]:
    validation = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    examples = [_mmlu_example(row) for row in validation]
    categories = list(dict.fromkeys(row["category"] for row in validation))
    test_by_category = defaultdict(list)
    for row in test:
        test_by_category[row["category"]].append(row)

    # Match the source implementation: append test examples in category-wise
    # chunks of five. With the default 210-example cap, this yields 15 items
    # per MMLU-Pro category across validation and test.
    offsets = {category: 0 for category in categories}
    while True:
        added = False
        for category in categories:
            start = offsets[category]
            chunk = test_by_category[category][start : start + 5]
            if not chunk:
                continue
            examples.extend(_mmlu_example(row) for row in chunk)
            offsets[category] += len(chunk)
            added = True
        if not added:
            break
    return examples


LOADERS = {
    "hex-phi": _load_hex_phi,
    "xstest": _load_xstest,
    "hh-rlhf": _load_hh_rlhf,
    "arc-challenge": _load_arc_challenge,
    "mmlu-pro": _load_mmlu_pro,
}


def load_benchmark(task: str) -> List[BenchmarkExample]:
    try:
        return LOADERS[task]()
    except KeyError as error:
        raise ValueError(f"Unsupported benchmark: {task}") from error
