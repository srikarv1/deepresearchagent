from adr.llm.openai_compat import chat_token_kwargs, is_reasoning_model


def test_gpt5_uses_max_completion_tokens_and_drops_zero_temperature():
    kwargs = chat_token_kwargs("gpt-5-mini", 800, 0.0)
    assert kwargs == {"max_completion_tokens": 800}
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("o3-mini")


def test_gpt4_keeps_max_tokens_and_temperature():
    kwargs = chat_token_kwargs("gpt-4.1", 800, 0.0)
    assert kwargs == {"max_tokens": 800, "temperature": 0.0}
    assert not is_reasoning_model("gpt-4.1")
