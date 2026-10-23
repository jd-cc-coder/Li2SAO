# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import re
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import tool sandbox functionality
from tool_sandbox import TOOL_CONFIGS, tool_registry

# One level of nested braces; enough for AIME integers and simple fractions.
_BOXED_CONTENT_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_ANSWER_BOXED_RE = re.compile(r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)


def _last_real_boxed(text: str) -> str | None:
    """Last \\boxed{...} that is not the invalid-action placeholder."""
    content = None
    for match in _BOXED_CONTENT_RE.finditer(text or ""):
        inner = match.group(1).strip()
        if inner and inner.lower() != "answer":
            content = inner
    return content

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use Python "
            "tools to solve mathematical problems. When you need "
            "to perform calculations, use the code_interpreter "
            "tool to execute code and get results."
        )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Stop on a real boxed answer (Answer: prefix optional). A lone \\boxed{73}
    # used to be treated as invalid, which injected a fake \\boxed{answer} and
    # made the strict DAPO grader read the hint instead of the real answer.
    answer_match = _ANSWER_BOXED_RE.search(prediction)
    if answer_match:
        content = answer_match.group(1).strip()
        if content and content.lower() != "answer":
            return "answer", content
    boxed = _last_real_boxed(prediction)
    if boxed is not None:
        return "answer", boxed

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Truncate after the last real boxed answer (Answer: prefix optional).
    last_box = None
    for match in _ANSWER_BOXED_RE.finditer(resp):
        if match.group(1).strip() and match.group(1).strip().lower() != "answer":
            last_box = match
    if last_box is None:
        for match in _BOXED_CONTENT_RE.finditer(resp):
            if match.group(1).strip() and match.group(1).strip().lower() != "answer":
                last_box = match
    if last_box is not None:
        return resp[: last_box.end()]

    return resp


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            # Semaphore lives in execute_tool; do not acquire it here or each
            # call takes two slots (halving tool_concurrency).
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to execute code, I should put the code between "
            "<code> and </code>. "
            "If I want to give the final answer, put it in a LaTeX boxed "
            "expression. Let me try again.\n"
        )
        done = False

    return next_obs, done


# Metadata keys for abort → resume (in-flight weight sync / partial rollout).
_RETOOL_PROMPT_LEN_KEY = "retool_prompt_len"
_RETOOL_TURN_KEY = "retool_turn"
_RETOOL_OPEN_TURN_TEXT_KEY = "retool_open_turn_text"


def _inflight_resume_enabled(args) -> bool:
    """True when this generate is running under in-flight abort→resubmit."""
    if getattr(args, "partial_rollout", False):
        return True
    if getattr(args, "inflight_inner_generate_function_path", None):
        return True
    rollout_fn = getattr(args, "rollout_function_path", "") or ""
    if "inflight_async" in rollout_fn:
        return True
    custom_fn = getattr(args, "custom_generate_function_path", "") or ""
    return "inflight_async.generate" in custom_fn


def _has_retool_prefix(sample: Sample) -> bool:
    tokens = getattr(sample, "tokens", None) or []
    return bool(tokens) and int(getattr(sample, "response_length", 0) or 0) > 0


def _should_resume_retool(args, sample: Sample, evaluation: bool = False) -> bool:
    """Resume a partial trajectory instead of wiping it.

    Sync colocate retries still start clean (the old behavior). Inflight /
    partial-rollout keeps tokens, log-probs, loss_mask, and weight_versions.
    """
    if evaluation:
        return False
    return _inflight_resume_enabled(args) and _has_retool_prefix(sample)


def _reset_sample_for_fresh_generate(sample: Sample) -> None:
    """Drop leftover rollout state so a new attempt does not concatenate."""
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.weight_versions = []
    sample.tool_call_count = 0
    sample.status = Sample.Status.PENDING
    meta = getattr(sample, "metadata", None)
    if meta is None:
        sample.metadata = {}
        meta = sample.metadata
    for key in (
        _RETOOL_PROMPT_LEN_KEY,
        _RETOOL_TURN_KEY,
        _RETOOL_OPEN_TURN_TEXT_KEY,
        "inflight_segment_tokens",
    ):
        meta.pop(key, None)


def _max_context_length(args) -> int:
    if getattr(args, "rollout_max_context_len", None) is not None:
        return int(args.rollout_max_context_len)
    return int(args.context_parallel_size) * int(args.max_tokens_per_gpu)


def _max_response_length(args, sampling_params: dict) -> int:
    return int(getattr(args, "rollout_max_response_len", None) or sampling_params.get("max_new_tokens") or 0)


def _per_turn_max_new_tokens(
    args,
    sampling_params: dict,
    *,
    total_length: int,
    response_length: int,
    max_context_length: int,
) -> int:
    """Clamp one HTTP generate by remaining context and remaining response."""
    remaining_context = max(0, int(max_context_length) - int(total_length))
    ori_max = _max_response_length(args, sampling_params)
    remaining_response = max(0, ori_max - int(response_length)) if ori_max > 0 else remaining_context
    requested = int(sampling_params.get("max_new_tokens", remaining_context) or remaining_context)
    return max(0, min(requested, remaining_context, remaining_response))


def _persist_retool_progress(
    sample: Sample,
    *,
    prompt_token_len: int,
    turn: int,
    tool_call_count: int,
    open_turn_text: str,
    response: str,
) -> None:
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata[_RETOOL_PROMPT_LEN_KEY] = int(prompt_token_len)
    sample.metadata[_RETOOL_TURN_KEY] = int(turn)
    sample.metadata[_RETOOL_OPEN_TURN_TEXT_KEY] = open_turn_text
    sample.tool_call_count = int(tool_call_count)
    sample.response = response


def _restore_retool_state(sample: Sample, state: GenerateState) -> tuple[list[int], list[int], str, int, int, str]:
    """Rebuild prompt / response / turn counters from a partial sample."""
    prompt_len = (sample.metadata or {}).get(_RETOOL_PROMPT_LEN_KEY)
    if prompt_len is None:
        prompt_len = max(0, len(sample.tokens) - int(sample.response_length))
    prompt_len = int(prompt_len)
    prompt_tokens_ids = list(sample.tokens[:prompt_len])
    response_token_ids = list(sample.tokens[prompt_len:])
    if len(response_token_ids) != int(sample.response_length):
        # Tokens and response_length drifted; trust tokens after the prompt.
        sample.response_length = len(response_token_ids)
    response = sample.response or ""
    if not response and response_token_ids:
        response = state.tokenizer.decode(response_token_ids)
        sample.response = response
    tool_call_count = int(getattr(sample, "tool_call_count", 0) or 0)
    start_turn = int((sample.metadata or {}).get(_RETOOL_TURN_KEY, 0) or 0)
    open_turn_text = (sample.metadata or {}).get(_RETOOL_OPEN_TURN_TEXT_KEY, "") or ""
    return prompt_tokens_ids, response_token_ids, response, tool_call_count, start_turn, open_turn_text


def _finalize_retool_sample(
    sample: Sample,
    *,
    prompt: str,
    prompt_tokens_ids: list[int],
    response_token_ids: list[int],
    response: str,
    tool_call_count: int,
    finish_type: str | None,
) -> Sample:
    sample.tokens = list(prompt_tokens_ids) + list(response_token_ids)
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response
    sample.tool_call_count = tool_call_count
    # Resume starts from ABORTED; map the terminal finish_reason once the
    # attempt actually ends. Do not overwrite TRUNCATED already set in-loop.
    if finish_type is not None and sample.status in (Sample.Status.PENDING, Sample.Status.ABORTED):
        match finish_type:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Multi-turn tool generate; supports in-flight abort → resume.

    Sync colocate still wipes leftover state and restarts (old behavior).
    Inflight / partial-rollout keeps tokens and continues the unfinished turn
    under the new policy weights. ``evaluation`` is accepted for the slime
    custom-generate signature.
    """
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    max_context_length = _max_context_length(args)
    inflight = _inflight_resume_enabled(args) and not evaluation
    resume = _should_resume_retool(args, sample, evaluation=evaluation)
    last_finish: str | None = None

    if resume:
        (
            prompt_tokens_ids,
            response_token_ids,
            response,
            tool_call_count,
            start_turn,
            open_turn_text,
        ) = _restore_retool_state(sample, state)
        if sample.status not in (Sample.Status.PENDING, Sample.Status.ABORTED):
            sample.status = Sample.Status.ABORTED if sample.response_length > 0 else Sample.Status.PENDING
    else:
        # Sync retries (and the first inflight attempt) start clean so leftover
        # tokens from a discarded attempt cannot concatenate into the next one.
        _reset_sample_for_fresh_generate(sample)
        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = list(prompt_tokens_ids)
        response = ""
        response_token_ids = []
        tool_call_count = 0
        start_turn = 0
        open_turn_text = ""
        _persist_retool_progress(
            sample,
            prompt_token_len=len(prompt_tokens_ids),
            turn=0,
            tool_call_count=0,
            open_turn_text="",
            response="",
        )

    # If abort cut the HTTP after a complete boxed / tool_call, finish that
    # turn before asking the engine for more tokens.
    if resume and open_turn_text:
        action, _ = postprocess_predictions(open_turn_text)
        if action in ("answer", "code"):
            next_obs, done = await execute_predictions(open_turn_text)
            if done:
                open_turn_text = ""
                sample.metadata[_RETOOL_OPEN_TURN_TEXT_KEY] = ""
                return _finalize_retool_sample(
                    sample,
                    prompt=prompt,
                    prompt_tokens_ids=prompt_tokens_ids,
                    response_token_ids=response_token_ids,
                    response=response,
                    tool_call_count=tool_call_count,
                    finish_type="stop",
                )
            if next_obs:
                if "<interpreter>" in next_obs:
                    tool_call_count += 1
                obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                overflow = (
                    len(prompt_tokens_ids) + len(response_token_ids) + len(obs_tokens_ids) - max_context_length
                )
                if overflow > 0:
                    obs_tokens_ids = obs_tokens_ids[: max(0, len(obs_tokens_ids) - overflow)]
                response_token_ids += obs_tokens_ids
                sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False)
                if overflow > 0:
                    response = state.tokenizer.decode(response_token_ids)
                    sample.status = Sample.Status.TRUNCATED
                    return _finalize_retool_sample(
                        sample,
                        prompt=prompt,
                        prompt_tokens_ids=prompt_tokens_ids,
                        response_token_ids=response_token_ids,
                        response=response,
                        tool_call_count=tool_call_count,
                        finish_type="length",
                    )
                response += next_obs
                open_turn_text = ""
                start_turn = start_turn + 1
                _persist_retool_progress(
                    sample,
                    prompt_token_len=len(prompt_tokens_ids),
                    turn=start_turn,
                    tool_call_count=tool_call_count,
                    open_turn_text="",
                    response=response,
                )

    for turn in range(start_turn, TOOL_CONFIGS["max_turns"]):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        remaining_budget = _per_turn_max_new_tokens(
            args,
            sampling_params,
            total_length=total_length,
            response_length=len(response_token_ids),
            max_context_length=max_context_length,
        )
        if remaining_budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        if inflight:
            try:
                from lisao.inflight_async.pause import get_pause_state

                pause_state = get_pause_state()
            except ImportError:
                pause_state = None
            if pause_state is not None and pause_state.is_paused():
                await pause_state.wait_until_resumed(
                    poll_s=float(getattr(args, "inflight_resubmit_wait", 0.5))
                )
                # Weight update landed between tool turns: return ABORTED so
                # the inflight wrapper records this prefix and resubmits.
                if sample.response_length > 0:
                    sample.status = Sample.Status.ABORTED
                    _persist_retool_progress(
                        sample,
                        prompt_token_len=len(prompt_tokens_ids),
                        turn=turn,
                        tool_call_count=tool_call_count,
                        open_turn_text=open_turn_text,
                        response=response,
                    )
                    return _finalize_retool_sample(
                        sample,
                        prompt=prompt,
                        prompt_tokens_ids=prompt_tokens_ids,
                        response_token_ids=response_token_ids,
                        response=response,
                        tool_call_count=tool_call_count,
                        finish_type="abort",
                    )

        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = remaining_budget
        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,
        }

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": len(tool_specs),
                        "debug/tools_used": response.count("<interpreter>"),
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass

        output = await post(url, payload)
        last_finish = output["meta_info"]["finish_reason"]["type"]

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            response += cur_response
            open_turn_text += cur_response
            response_token_ids += cur_response_token_ids
            sample.append_response_tokens(
                args,
                tokens=cur_response_token_ids,
                log_probs=cur_log_probs,
                trainable=True,
                meta_info=output["meta_info"],
            )
        elif last_finish == "abort":
            # Keep prefix; inflight wrapper will resubmit this sample.
            sample.status = Sample.Status.ABORTED
            _persist_retool_progress(
                sample,
                prompt_token_len=len(prompt_tokens_ids),
                turn=turn,
                tool_call_count=tool_call_count,
                open_turn_text=open_turn_text,
                response=response,
            )
            return _finalize_retool_sample(
                sample,
                prompt=prompt,
                prompt_tokens_ids=prompt_tokens_ids,
                response_token_ids=response_token_ids,
                response=response,
                tool_call_count=tool_call_count,
                finish_type="abort",
            )
        else:
            # Text but no per-token logprobs — cannot train on this turn.
            sample.status = Sample.Status.ABORTED
            _persist_retool_progress(
                sample,
                prompt_token_len=len(prompt_tokens_ids),
                turn=turn,
                tool_call_count=tool_call_count,
                open_turn_text=open_turn_text,
                response=response,
            )
            return _finalize_retool_sample(
                sample,
                prompt=prompt,
                prompt_tokens_ids=prompt_tokens_ids,
                response_token_ids=response_token_ids,
                response=response,
                tool_call_count=tool_call_count,
                finish_type="abort",
            )

        _persist_retool_progress(
            sample,
            prompt_token_len=len(prompt_tokens_ids),
            turn=turn,
            tool_call_count=tool_call_count,
            open_turn_text=open_turn_text,
            response=response,
        )

        if last_finish == "abort":
            # Tokens from this HTTP are kept; wrapper retries the same turn.
            sample.status = Sample.Status.ABORTED
            return _finalize_retool_sample(
                sample,
                prompt=prompt,
                prompt_tokens_ids=prompt_tokens_ids,
                response_token_ids=response_token_ids,
                response=response,
                tool_call_count=tool_call_count,
                finish_type="abort",
            )

        if last_finish == "length":
            break

        # Parse the whole open turn, not just this HTTP chunk, so a tool_call
        # split across an abort→resume still executes.
        next_obs, done = await execute_predictions(open_turn_text)
        if done:
            open_turn_text = ""
            sample.metadata[_RETOOL_OPEN_TURN_TEXT_KEY] = ""
            break

        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        overflow = len(prompt_tokens_ids) + len(response_token_ids) + len(obs_tokens_ids) - max_context_length
        truncated_by_observation = overflow > 0
        if truncated_by_observation:
            obs_tokens_ids = obs_tokens_ids[: max(0, len(obs_tokens_ids) - overflow)]

        response_token_ids += obs_tokens_ids
        sample.append_response_tokens(args, tokens=obs_tokens_ids, trainable=False)

        if sample.rollout_log_probs is not None:
            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if truncated_by_observation:
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            open_turn_text = ""
            break
        response += next_obs
        open_turn_text = ""
        _persist_retool_progress(
            sample,
            prompt_token_len=len(prompt_tokens_ids),
            turn=turn + 1,
            tool_call_count=tool_call_count,
            open_turn_text="",
            response=response,
        )

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    return _finalize_retool_sample(
        sample,
        prompt=prompt,
        prompt_tokens_ids=prompt_tokens_ids,
        response_token_ids=response_token_ids,
        response=response,
        tool_call_count=tool_call_count,
        finish_type=last_finish,
    )


async def reward_func(args, sample, **kwargs):
    """Same 0/1 flexible math RM as examples/lisao (no DAPO −1 / tool shaping)."""
    # Prefer the LiSAO-path helper so train and eval share strip + grade.
    try:
        from eval_rm import grade_one
    except ImportError:
        from lisao.eval.math_rm import grade_math_flexible_dict

        if not isinstance(sample, Sample):
            raise TypeError("Sample must be an instance of Sample class.")
        return grade_math_flexible_dict(sample.response or "", sample.label)
    del args, kwargs
    if isinstance(sample, list):
        return [grade_one(s) for s in sample]
    return grade_one(sample)
