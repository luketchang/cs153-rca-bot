from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from oncall.constants import MODEL_SEED

load_dotenv()


def get_llm(model: str, max_retries: int = 3, timeout: int = 60, **extra_kwargs):
    # These initial kwargs should always be added and need to apply to both ChatOpenAI and ChatAnthropic.
    kwargs = {
        "model": model,
        # "temperature": MODEL_TEMPERATURE,
    }
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if timeout is not None:
        kwargs["timeout"] = timeout

    kwargs.update(extra_kwargs)

    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        if not kwargs.get("seed"):
            kwargs["seed"] = MODEL_SEED
        if model.startswith("o3"):
            kwargs["reasoning_effort"] = "high"
            kwargs["disabled_params"] = {"parallel_tool_calls": None}
        return ChatOpenAI(**kwargs)
    elif model.startswith("claude"):
        return ChatAnthropic(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model}")
        raise ValueError(f"Unsupported model: {model}")
