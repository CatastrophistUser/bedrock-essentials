# Converse API waits until the whole response is generated and then outputs the full block

import boto3

# Configuring the runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")

# Response Body
response = bedrock_runtime.converse(
    modelId="minimax.minimax-m2.5",
    system=[{"text": "You are a concise AWS tutor."}],
    messages=[
        {"role": "user", "content": [{"text": "Explain Amazon Bedrock and its use cases"}]}
    ],

    # Inference Parameters
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0.8,
        "topP": 0.9,
    },
)

# Join Response and Print Statement
text = "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
print(text)
print(response["stopReason"])