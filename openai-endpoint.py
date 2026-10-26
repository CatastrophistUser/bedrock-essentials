# Bedrock's OpenAI-compatible endpoint (bedrock-mantle): use the normal OpenAI SDK, just pointed at Bedrock
# bedrock-mantle is a Bedrock endpoint that accepts the same API as OpenAI.

import os
from openai import OpenAI
from aws_bedrock_token_generator import provide_token

# Add AWS_BEARER_TOKEN_BEDROCK 
api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or provide_token(region="eu-central-1")

client = OpenAI(
    base_url="https://bedrock-mantle.eu-central-1.api.aws/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="openai.gpt-oss-120b",
    messages=[{"role": "user", "content": "One fact about Scion by Anthropic."}],
    max_completion_tokens=512,
)
print(response.choices[0].message.content)
print(response.usage)
