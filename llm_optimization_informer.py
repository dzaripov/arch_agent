import os
import json
import logging
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from informer_tool import run_informer_experiment, informer_tools

# --- 1. SETUP ---
logging.basicConfig(
    filename='logs_history.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def init_optuna_db(db_path='optuna_llm_logs.db'):
    """Инициализация SQLite базы данных для хранения историй экспериментов"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Создание таблицы trials (как в Optuna)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER,
            x REAL,
            y REAL,
            loss REAL,
            timestamp TEXT,
            llm_response TEXT
        )
    ''')
    
    # Создание таблицы для хранения истории сообщений
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Optuna-compatible database initialized at {db_path}")

init_optuna_db()

def log_trial_to_optuna(iteration, x, y, loss, llm_response="", db_path='optuna_llm_logs.db'):
    """Логирование trial в формате Optuna"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO trials (iteration, x, y, loss, timestamp, llm_response)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (iteration, x, y, loss, datetime.now().isoformat(), llm_response))
    
    conn.commit()
    conn.close()
    logging.info(f"Trial logged to Optuna DB: iteration={iteration}, x={x}, y={y}, loss={loss}")

def log_message_to_history(role, content, db_path='optuna_llm_logs.db'):
    """Логирование сообщений в историю разговора"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO conversation_history (timestamp, role, content)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), role, content))
    
    conn.commit()
    conn.close()


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
    {"role": "user", "content": "Your goal is to find the best hyperparameter values for the Informer model to minimize MSE on the ETDataset (Electricity Transformer Dataset) for time-series forecasting task.\n**DATASET DETAILS** The dataset columns include information on the date (recorded date), HUFL (High UseFul Load), HULL (High UseLess Load), MUFL (Middle UseFul Load), MULL (Middle UseLess Load), LUFL (Low UseFul Load), LULL (Low UseLess Load), and a target column OT (Oil Temperature). "
    "You have a function that runs a long sequence time-series forecasting experiment using the Informer model. "
    # "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. "
    # Аналогично оптуне:
    "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. Here's the range of hyperparameters for reference, but you can use any values in between these ranges: seq_len_choices: Sequence[int] = (48, 96, 168, 336, 720); label_len_choices: Sequence[int] = (24, 48, 96, 168, 336); pred_len_choices: Sequence[int] = (24, 48, 96, 168, 336, 720); d_model_choices: Sequence[int] = (256, 512); n_heads_choices: Sequence[int] = (4, 8); e_layers_choices: Sequence[int] = (2, 3); d_layers_choices: Sequence[int] = (1, 2); d_ff_choices: Sequence[int] = (512, 1024, 2048); factor_choices: Sequence[int] = (1, 3, 5); dropout: tuple[float, float] = (0.01, 0.3); learning_rate: tuple[float, float] = (1e-5, 5e-4); batch_size_choices: Sequence[int] = (16, 32, 64); train_epochs: tuple[int, int] = (4, 20); patience: tuple[int, int] = (2, 6); s_layers_choices: Sequence[str] = ('3,2,1', '4,3,2'); attn_choices: Sequence[str] = ('prob', 'full'); embed_choices: Sequence[str] = ('timeF', 'fixed', 'learned'); activation_choices: Sequence[str] = ('gelu', 'relu'); distil_options: Sequence[bool] = (True, False); output_attention_options: Sequence[bool] = (False, True); mix_options: Sequence[bool] = (True, False); padding_options: Sequence[int] = (0, 1); lradj_choices: Sequence[str] = ('type1', 'type2')"
    f"Use both exploration and exploitation methods. You have {ITERATIONS} iterations. "}
]


# Альтернатива
# messages = [
#     {"role": "user", "content": "Your goal is to find the best hyperparameter values for the Informer model to minimize MSE on the ETDataset (Electricity Transformer Dataset) for time-series forecasting task.\n**DATASET DETAILS** The dataset columns include information on the date (recorded date), HUFL (High UseFul Load), HULL (High UseLess Load), MUFL (Middle UseFul Load), MULL (Middle UseLess Load), LUFL (Low UseFul Load), LULL (Low UseLess Load), and a target column OT (Oil Temperature). "
#     "You have a function that runs a long sequence time-series forecasting experiment using the Informer model. "
#     "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. **Hyperparameters** you may want to consider optimizing: learning_rate in range from 0.00001 to 0.01, explore both lower and higher values; number of encoder layers (e_layers) in range from 1 to 6, more layers = more capacity but longer training; any other hyperparameters listed in informer_tools. **SEARCH GUIDELINES:** - Typical good learning rates for Informer: 0.00005, 0.0001, 0.0005, 0.001; - Encoder layers: 1-4 often work well for this dataset size; - If MSE > 1.0: try different regions (learning rate too high/low, layers inappropriate); - If MSE < 0.3: fine-tune around current values; - Consider training time vs performance trade-off. "
#     f"Use both exploration and exploitation methods. You have {ITERATIONS} iterations. "}
# ]

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

        log_message_to_history("assistant", str(response_message))

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

            # формат Optuna
            if 'x' in function_args and 'y' in function_args:
                x = function_args['x']
                y = function_args['y']
                # NOTEFORME чекнуть что лосс именно тут
                loss = function_response.get('loss', function_response.get('val_loss', 0))
                log_trial_to_optuna(
                    iteration=i+1, 
                    x=x, 
                    y=y, 
                    loss=loss, 
                    llm_response=json.dumps(function_response)
                )

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

            log_message_to_history("tool", json.dumps(function_response))
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

    final_prompt = (
        "Based on the history of our conversation, what are the best values for x and y you found, "
        "and what was the minimum loss? Summarize the results."
    )
    messages.append({
        "role": "user",
        "content": final_prompt
    })

    log_message_to_history("user", final_prompt)

    final_summary_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    logging.info(f"Final summary from LLM: {final_summary_response.choices[0].message.content}")
    print("\n✅ Final Report from the Agent:")
    print(final_summary_response.choices[0].message.content)
    log_message_to_history("assistant", final_summary_response.choices[0].message.content)
except Exception as e:
    logging.error(f"Error during final summary request: {e}")
    print(f"Error during final summary request: {e}")


# --- 4. Optuna dashboard ---

def analyze_results():
    """Анализ результатов после завершения оптимизации"""
    print("\n ====== Analysis Started ======")
    
    try:
        # Загрузка данных
        conn = sqlite3.connect('optuna_llm_logs.db')
        trials_df = pd.read_sql_query("SELECT * FROM trials ORDER BY loss ASC", conn)
        conn.close()
        
        if not trials_df.empty:
            print(f"Best result: x={trials_df.iloc[0]['x']}, y={trials_df.iloc[0]['y']}, loss={trials_df.iloc[0]['loss']}")
            print(f"Total trials: {len(trials_df)}")
            
            # Создание простой визуализации
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(trials_df['iteration'], trials_df['loss'], 'o-')
            plt.title('Optimization Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.scatter(trials_df['x'], trials_df['y'], c=trials_df['loss'], cmap='viridis')
            plt.colorbar()
            plt.title('Parameter Space')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plt.tight_layout()
            plt.savefig('optimization_analysis.png', dpi=300)
            plt.show()
            
    except ImportError as e:
        print(f"Analysis libraries not available: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

try:
    analyze_results()
except Exception as e:
    logging.error(f"Error during analyzis: {e}")
    print(f"Error during analyzis: {e}")
