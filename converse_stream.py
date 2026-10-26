# Converse Stream streams each token as it is generated

import boto3

# Configuring the runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")

# Request a streamed response (tokens arrive as they are generated)
stream = bedrock_runtime.converse_stream(
    modelId="eu.amazon.nova-2-lite-v1:0",
    messages=[{"role": "user", "content": [{"text": "Write a haiku about AWS EBS."}]}],
    inferenceConfig={"maxTokens": 200},
)

# Read the events one by one as they arrive
for event in stream["stream"]:
    # A chunk of the answer text, flush is mandatory for streaming
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"].get("text", ""), end="", flush=True)
    # The model finished: why it stopped
    elif "messageStop" in event:
        print("\nstop:", event["messageStop"]["stopReason"])
    # Last event: token counts for the whole call
    elif "metadata" in event:
        print("usage:", event["metadata"]["usage"])
