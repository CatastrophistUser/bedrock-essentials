# InvokeModel sends the provider's own (native) request body instead of the unified Converse format

import json
import boto3

# Configuring the runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")

# OpenAI gpt-oss: the body is the OpenAI Chat Completions format
body = {
    "messages": [{"role": "user", "content": "List 3 S3 storage classes."}],
    "max_completion_tokens": 512,
}
response = bedrock_runtime.invoke_model(modelId="openai.gpt-oss-120b-1:0", body=json.dumps(body))

# The response body is a stream: read it, then parse the JSON
data = json.loads(response["body"].read())

# gpt-oss puts its thinking in <reasoning>...</reasoning> before the answer
text = data["choices"][0]["message"]["content"]
print(text.split("</reasoning>")[-1].strip())
print(data["usage"])

# Embeddings use InvokeModel too (Titan's own body format)
emb = bedrock_runtime.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "vector me", "dimensions": 512, "normalize": True}),
)
vector = json.loads(emb["body"].read())["embedding"]
print(len(vector), vector[:5])
