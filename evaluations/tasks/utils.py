"""Code scoring helpers for the custom lm-eval tasks."""

from lm_eval.tasks.mbpp.utils import pass_at_k


_STOP_SEQUENCES = (
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n# Test",
    "\nassert ",
    "\n```",
)


def _strip_stop_prefix_suffix(completion: str) -> str:
    # TODO: Revisit partial-stop stripping: it can remove actual code or repair
    # truncated output, changing pass@1. Audit raw vs. stripped predictions
    # before changing this policy, and rescore baselines and speculative runs alike.
    suffix_length = max(
        (
            length
            for stop in _STOP_SEQUENCES
            for length in range(1, len(stop) + 1)
            if completion.endswith(stop[:length])
        ),
        default=0,
    )
    return completion[:-suffix_length] if suffix_length else completion


def pass_at_1(
    references: list[str], predictions: list[str] | list[list[str]]
) -> float:
    if isinstance(predictions[0], str):
        predictions = [[prediction] for prediction in predictions]
    predictions = [
        [_strip_stop_prefix_suffix(completion) for completion in candidates]
        for candidates in predictions
    ]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
        timeout=5,
    )[0]["pass@1"]
