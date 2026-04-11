export ANTHROPIC_BASE_URL=http://35.220.164.252:3888
export ANTHROPIC_AUTH_TOKEN="sk-yjKTOBMpL9Ee8pFnzpXRWZY7840d42kLDbdcanJoOFBXa9ha"
export API_TIMEOUT_MS=600000
export ANTHROPIC_MODEL=claude-sonnet-4-6-thinking
export ANTHROPIC_SMALL_FAST_MODEL=claude-sonnet-4-6-thinking
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
source <(curl -sSL http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh)