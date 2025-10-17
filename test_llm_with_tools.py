import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()

provider = 'local'
ITERATIONS = 30

if provider == 'openrouter':
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    model = "deepseek/deepseek-chat-v3.1"
elif provider == 'local':
    client = OpenAI(
        base_url="http://0.0.0.0:30000/v1",
        api_key="None",
    )

    model = "Qwen/Qwen3-4B-Instruct-2507-FP8"


print(f'Using model: {model} by {provider}')

# --- 2. DEFINE THE TOOL (SCHEMA AND IMPLEMENTATION) ---

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_loss",
            "description": (
                "Calculates the loss for a given set of parameters x and y. "
                "The goal is to find the values of x and y that MINIMIZE this loss. "
                "Provide your best guess for the next x and y to try."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "The first parameter (a float)."},
                    "y": {"type": "number", "description": "The second parameter (a float)."}
                },
                "required": ["x", "y"]
            }
        }
    }
]

def get_loss(x: float, y: float) -> float:
    """Calculates the loss of x and y"""
    print(f"--- Executing get_loss with x={x}, y={y} ---")
    loss = (x - 3) ** 2 + (y - 5) ** 2
    print(f"--- Loss calculated: {loss} ---")

    # here we need to log hyperparameters to visualize trajectory of optimization
    return loss

available_functions = {
    "get_loss": get_loss,
}

# --- 3. THE OPTIMIZATION LOOP ---

messages = [
    {"role": "user", "content": "Your goal is to find the values for x and y that minimize a secret loss function. "
    "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. "
    f"Use both exploration and exploitation methods. You have {ITERATIONS} iterations. "
     "Values range of x and y are in [-10, 10]"}
]

for i in range(ITERATIONS):
    print(f"\n ====== Iteration {i + 1}/{ITERATIONS} ======")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    # print(response_message)
    
    messages.append(response_message)

    if not response_message.tool_calls:
        print("LLM finished or decided not to call a tool. Stopping loop.")
        break
        # ask to call tool again

    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"LLM suggests trying: x={function_args.get('x')}, y={function_args.get('y')}")
        
        function_response = function_to_call(**function_args)

        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps({"loss": function_response}),
            }
        )

print("\n ====== Optimization Finished ======")

# --- 4. FINAL REFLECTION ---

print("\nAsking LLM for a final summary...")
messages.append({
    "role": "user",
    "content": "Based on the history of our conversation, what are the best values for x and y you found, and what was the minimum loss? Summarize the results."
})

final_summary_response = client.chat.completions.create(
    model=model,
    messages=messages
)

print("\n✅ Final Report from the Agent:")
print(final_summary_response.choices[0].message.content)