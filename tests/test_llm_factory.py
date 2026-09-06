from adr.llm.factory import build_llm


def test_azure_foundry_provider_uses_azure_defaults(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-key")

    llm = build_llm(
        {
            "provider": "azure_foundry",
            "model": "research-deployment",
            "endpoint": "https://resource.services.ai.azure.com",
        }
    )

    assert llm.model == "research-deployment"
    assert str(llm._client.base_url) == "https://resource.services.ai.azure.com/openai/v1/"
    assert llm._client.api_key == "azure-test-key"