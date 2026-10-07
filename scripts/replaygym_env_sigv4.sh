# ReplayGym overlay for IAM (SigV4) credentials: source after .env.
# Bedrock bearer keys are blocked for this account, so every OpenAI-style call
# goes to scripts/bedrock_openai_shim.py on :4000, which signs with the IAM keys.
OPENAI_API_KEY="local-shim"
OPENAI_BASE_URL="http://127.0.0.1:4000/v1"
REPLAY_AGENT_CONFIG="configs/agents/gpt_researcher_gym_bedrock_sigv4.yaml"
REPLAY_JUDGE_MODEL="qwen.qwen3-32b"
REPLAY_EMBED="local"
REPLAY_TAG="bedrock"
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
LLM_KWARGS='{"timeout": 600, "max_retries": 3}'
