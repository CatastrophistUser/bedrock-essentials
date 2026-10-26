import boto3

# Configuring the runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")

# Response Body
response = bedrock_runtime.converse(
    modelId="eu.amazon.nova-micro-v1:0",
    system=[
        # System prompt is made longer (* 200) else caching is skipped for small tokens
        {"text": "You are a concise AWS tutor. " + "Amazon Bedrock is a managed service for foundation models. " * 200},

        # Caching after a long, stable prefix (system prompt, reference document, tool list)
        {"cachePoint": {"type": "default"}},
    ],
    messages=[
        {"role": "user", "content": [{"text": "Explain Amazon Bedrock and its use cases"}]}
    ],
    inferenceConfig={"maxTokens": 300},
)

# Later calls with the same prefix read it from the cache
print("Read Input Cache:", response["usage"].get("cacheReadInputTokens"))
print("Write Input Cache:", response["usage"].get("cacheWriteInputTokens"))
print(response["output"]["message"]["content"][0]["text"])
