# Source after .env: route every OpenAI-SDK call (agent, writer, DRG judge) to Bedrock.
#   set -a; source .env; source scripts/replaygym_env.sh; set +a
OPENAI_API_KEY="$AWS_BEARER_TOKEN_BEDROCK"
OPENAI_BASE_URL="https://bedrock-mantle.${AWS_REGION:-us-east-1}.api.aws/v1"
REPLAY_AGENT_CONFIG="configs/agents/gpt_researcher_gym_bedrock.yaml"
REPLAY_JUDGE_MODEL="qwen.qwen3-32b"
REPLAY_EMBED="local"
REPLAY_TAG="bedrock"
# Local HF embeddings are called from several agent threads at once; without
# these, torch/tokenizers can deadlock (spinning threads, no progress).
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
# Bedrock connections occasionally stall forever; cap every LLM call (agent,
# writer) at 5 min so the fork's own retry kicks in instead of hanging a run.
LLM_KWARGS='{"timeout": 300, "max_retries": 2}'
