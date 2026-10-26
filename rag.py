# RAG with Knowledge Bases: Bedrock reads your documents, chunks + embeds them into a vector store,
# and answers questions using the most relevant chunks (with citations)

import sys
import time
import boto3

REGION = "eu-central-1"

# Fill these in after creating the Knowledge Base in the console (Section 1)
KB_ID, DS_ID = "KBXXXXXXXX", "DSXXXXXXXX"

if KB_ID.startswith("KBXXX"):
    sys.exit("Set KB_ID and DS_ID first (Bedrock console > Knowledge Bases > your KB).")

agent = boto3.client("bedrock-agent", region_name=REGION)              # manage KBs + ingestion
kbr = boto3.client("bedrock-agent-runtime", region_name=REGION)        # query KBs
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)  # call models

question = "What is our parental leave policy?"


# ---------------------------------------------------------------------------
# Section 2: Sync the data source (read documents from S3 -> chunk -> embed -> store)
# ---------------------------------------------------------------------------
job = agent.start_ingestion_job(knowledgeBaseId=KB_ID, dataSourceId=DS_ID)["ingestionJob"]
while job["status"] in ("STARTING", "IN_PROGRESS", "STOPPING"):
    time.sleep(10)
    job = agent.get_ingestion_job(knowledgeBaseId=KB_ID, dataSourceId=DS_ID,
                                  ingestionJobId=job["ingestionJobId"])["ingestionJob"]
print(job["status"], job.get("statistics"))


# ---------------------------------------------------------------------------
# Section 3a: Retrieve and generate in one call
# ---------------------------------------------------------------------------
# Inference profiles need your account ID in the ARN, so look it up instead of hardcoding it
account_id = boto3.client("sts").get_caller_identity()["Account"]
model_arn = f"arn:aws:bedrock:{REGION}:{account_id}:inference-profile/eu.amazon.nova-2-lite-v1:0"

response = kbr.retrieve_and_generate(
    input={"text": question},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": model_arn,
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 5}},
        },
    },
)
print(response["output"]["text"])

# Citations: which source documents each part of the answer came from
for c in response["citations"]:
    for ref in c["retrievedReferences"]:
        print("-", ref["location"].get("s3Location", {}).get("uri", ref["location"]))


# ---------------------------------------------------------------------------
# Section 3b: Retrieve only, then call Converse yourself (control over prompt, model, streaming)
# ---------------------------------------------------------------------------
hits = kbr.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": question},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)["retrievalResults"]

# Number each chunk so the model can cite it like [2]
context = "\n\n".join(f"[{i+1}] {h['content'].get('text', '')}" for i, h in enumerate(hits))

answer = bedrock_runtime.converse(
    modelId="eu.amazon.nova-2-lite-v1:0",
    system=[{"text": "Answer only from the numbered context. Cite like [2]. Say 'not found' if absent."}],
    messages=[{"role": "user", "content": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}]}],
    inferenceConfig={"maxTokens": 600, "temperature": 0.1},
)
print("".join(b["text"] for b in answer["output"]["message"]["content"] if "text" in b))
