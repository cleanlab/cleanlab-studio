# Test TLM
TEST_PROMPT = "What is the capital of France?"
TEST_RESPONSE = "Paris"
TEST_PROMPT_BATCH = ["What is the capital of France?", "What is the capital of Ukraine?"]
TEST_RESPONSE_BATCH = ["Paris", "Kyiv"]

# Test validation tests for TLM
MAX_PROMPT_LENGTH_TOKENS: int = 70_000
MAX_RESPONSE_LENGTH_TOKENS: int = 15_000
MAX_COMBINED_LENGTH_TOKENS: int = 70_000

CHARACTERS_PER_TOKEN: int = 4
