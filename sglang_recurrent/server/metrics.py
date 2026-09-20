"""Small, stable generation metrics emitted by both scheduler paths."""

import json
from collections.abc import Mapping
from logging import Logger


def log_decode_metrics(
    logger: Logger,
    decode_tokens: int,
    decode_seconds: float,
    loopspec_stats: Mapping[str, int] | None = None,
) -> None:
    """Emit one generation_metrics JSON log.

    decode_tokens and decode_seconds describe the interval after the first
    completion token; callers measure that interval. loopspec_stats optionally
    supplies the LoopSpec executor's integer counters (None for baseline).
    Counter consistency is checked by the offline result parser, not here.
    """
    metrics: dict[str, int | float | None] = {
        "decode_seconds": decode_seconds,
        "decode_tokens": decode_tokens,
        # No generated decode tokens or a nonpositive duration means no rate.
        # None is serialized as JSON null rather than a misleading zero rate.
        "decode_tok_s": (
            decode_tokens / decode_seconds
            if decode_tokens and decode_seconds > 0
            else None
        ),
    }

    if loopspec_stats is not None:
        # First-stage acceptances are all verifications minus rejections.
        # A first-stage rejection skips q2 when no second proposal is verified;
        # otherwise it contributes to second_accept or second_reject.
        metrics.update(
            pipeline_cycles=loopspec_stats["pipeline_cycles"],
            first_accept=loopspec_stats["verified_proposals"] - loopspec_stats["first_rejections"],
            first_reject=loopspec_stats["first_rejections"],
            second_skip=loopspec_stats["first_rejections"] - loopspec_stats["second_verifications"],
            second_accept=loopspec_stats["second_acceptances"],
            second_reject=loopspec_stats["second_rejections"],
        )

    # The result parser locates records by this prefix and reads the JSON body.
    logger.info(
        "generation_metrics %s",
        json.dumps(metrics, separators=(",", ":"), sort_keys=True),
    )
