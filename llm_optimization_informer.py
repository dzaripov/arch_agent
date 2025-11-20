import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from informer_tool import run_informer_experiment, informer_tools

# --- 1. SETUP ---
logging.basicConfig(
    filename='logs_history.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

provider = 'openrouter'
ITERATIONS = 10

try:
    if provider == 'openrouter':
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        model = "alibaba/tongyi-deepresearch-30b-a3b:free"
    elif provider == 'local':
        client = OpenAI(
            base_url="http://0.0.0.0:30000/v1",
            api_key="None",
        )

        model = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    logging.info(f'Using model: {model} by {provider}')
    print(f'Using model: {model} by {provider}')

except Exception as e:
    logging.error(f"Error during setup: {e}")
    raise

# --- 2. DEFINE THE TOOL (SCHEMA AND IMPLEMENTATION) ---


available_functions = {
    "run_informer_experiment": run_informer_experiment,
}

# --- 3. THE OPTIMIZATION LOOP ---

messages = [
    {"role": "user", "content": "Your goal is to find best hyperparameter values for Informer model. "
    "You have a fucntions that runs a long sequence time-series forecasting experiment using the Informer model. "
    "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. "
    f"Use both exploration and exploitation methods. You have {ITERATIONS} iterations. "}
]

for i in range(ITERATIONS):
    try:
        print(f"\n ====== Iteration {i + 1}/{ITERATIONS} ======")
        logging.info(f"Starting iteration {i + 1}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=informer_tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        # print(response_message)
        
        messages.append(response_message)

        if not response_message.tool_calls:
            logging.info("LLM finished or decided not to call a tool. Stopping loop.")
            print("LLM finished or decided not to call a tool. Stopping loop.")
            break
            # maybe ask to call tool again if llm tries to call tool else break

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            if not function_to_call:
                logging.warning(f"Function '{function_name}' not found.")
                continue

            function_args = json.loads(tool_call.function.arguments)
            logging.info(f"LLM suggests trying: {function_args}")
            print(f"LLM suggests trying: {function_args}")
        
            function_response = function_to_call(**function_args)

            logging.info(f"Results: {function_response}")
            print(f'Results: {function_response}')

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error on iteration {i+1}: {e}")
        print(f"JSON decode error: {e}")
    except Exception as e:
        logging.error(f"Error during iteration {i+1}: {e}")
        print(f"Error during iteration {i+1}: {e}")

logging.info("Optimization loop completed.")
print("\n ====== Optimization Finished ======")

# --- 4. FINAL REFLECTION ---
try:
    logging.info("Requesting final summary from LLM.")
    print("\nAsking LLM for a final summary...")
    messages.append({
        "role": "user",
        "content": "Based on the history of our conversation, what are the best values for x and y you found, and what was the minimum loss? Summarize the results."
    })

    final_summary_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    logging.info(f"Final summary from LLM: {final_summary_response.choices[0].message.content}")
    print("\n✅ Final Report from the Agent:")
    print(final_summary_response.choices[0].message.content)
    
except Exception as e:
    logging.error(f"Error during final summary request: {e}")
    print(f"Error during final summary request: {e}")
