# describe tools with a name, description and JSON Schema in toolConfig. 
# The model never runs anything itself: it replies with stopReason = tool_use and a toolUse block; your code runs the function and sends a toolResult back

import boto3

# Configuring the runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-central-1")
MODEL = "eu.amazon.nova-2-lite-v1:0"


# The actual tool: a normal Python function
def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 31, "sky": "haze"}  


# Map tool names (as the model sees them) to Python functions
TOOLS = {"get_weather": get_weather}

# Describe the tool to the model: name, what it does, and its inputs (JSON Schema)
tool_config = {
    "tools": [{
        "toolSpec": {
            "name": "get_weather",
            "description": "Current weather for a city. Returns temperature in Celsius and sky conditions.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            }},
        }
    }],
    # "toolChoice": {"auto": {}}   # or {"any": {}} / {"tool": {"name": "get_weather"}}
}

messages = [{"role": "user", "content": [{"text": "Do I need an umbrella in Bombay today?"}]}]

# Keep calling the model until it stops asking for tools
while True:
    response = bedrock_runtime.converse(modelId=MODEL, messages=messages, toolConfig=tool_config,
                                        inferenceConfig={"maxTokens": 1024})
    msg = response["output"]["message"]
    messages.append(msg)

    # No tool requested: this is the final answer
    if response["stopReason"] != "tool_use":
        print("".join(b["text"] for b in msg["content"] if "text" in b))
        break

    # Run every tool the model asked for and collect the results
    results = []
    for block in msg["content"]:
        if "toolUse" in block:
            call = block["toolUse"]
            print("Model called:", call["name"], call["input"])
            try:
                out = TOOLS[call["name"]](**call["input"])
                results.append({"toolResult": {"toolUseId": call["toolUseId"], "content": [{"json": out}]}})
            except Exception as e:
                results.append({"toolResult": {"toolUseId": call["toolUseId"],
                                               "content": [{"text": str(e)}], "status": "error"}})

    # Tool results go back to the model as a "user" message
    messages.append({"role": "user", "content": results})
