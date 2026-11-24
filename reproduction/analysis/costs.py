"""Shared token accounting for saved responses and request ledgers."""


def token_cost(usage, input_rate, output_rate, cached_input_rate=None):
    prompt = usage["prompt_tokens"]
    completion = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if not 0 <= cached <= prompt:
        raise ValueError("Cached input token count must be between zero and prompt tokens")
    cached_rate = input_rate if cached_input_rate is None else cached_input_rate
    return (prompt - cached) * float(input_rate) + cached * float(cached_rate) + completion * float(output_rate)


def response_usage(row, role):
    """Use the final response's cache counts, excluding earlier retry responses."""
    prefix = "grader" if role == "judge" else role
    usage = {"prompt_tokens": row[prefix + "_prompt_tokens"],
             "completion_tokens": row[prefix + "_completion_tokens"]}
    for call in reversed(row.get("api_calls", [])):
        observed = call.get("usage") or {}
        if (call["stage"] == role and call["status"] == "succeeded"
                and all(observed.get(key) == value for key, value in usage.items())):
            return observed
    return usage
