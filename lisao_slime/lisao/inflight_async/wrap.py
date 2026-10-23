"""Install the In-Flight abort→resubmit generate wrapper.

Kept free of slime/sglang imports so unit tests can cover wrapping without
torch. :mod:`lisao.inflight_async.rollout` calls this before starting workers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lisao.inflight_async.wrap")

CUSTOM_GENERATE_PATH = "lisao.inflight_async.generate.generate"


def ensure_inflight_generate(args: Any) -> None:
    """Wrap ``args.custom_generate_function_path`` with abort→resubmit generate.

    If the user already set a custom generate (e.g. ReTool multi-turn), keep it
    as ``inflight_inner_generate_function_path`` and point
    ``custom_generate_function_path`` at the inflight wrapper. Calling twice
    is a no-op once the wrapper is installed.
    """
    current = getattr(args, "custom_generate_function_path", None)
    if current is None:
        args.custom_generate_function_path = CUSTOM_GENERATE_PATH
        logger.info("inflight-async: using custom generate %s", CUSTOM_GENERATE_PATH)
        return
    if current == CUSTOM_GENERATE_PATH:
        return
    existing_inner = getattr(args, "inflight_inner_generate_function_path", None)
    if existing_inner and existing_inner != current:
        logger.info(
            "inflight-async: keeping inner generate %s (not replacing with %s)",
            existing_inner,
            current,
        )
    else:
        args.inflight_inner_generate_function_path = current
        logger.info(
            "inflight-async: wrapping user custom generate %s with %s",
            current,
            CUSTOM_GENERATE_PATH,
        )
    args.custom_generate_function_path = CUSTOM_GENERATE_PATH
