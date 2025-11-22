import os
import json
import logging
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from informer_tool import run_informer_experiment, informer_tools

# --- 1. SETUP ---
logging.basicConfig(
    filename="logs_history.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def init_optuna_db(db_path="optuna_llm_logs.db"):
    """Инициализация SQLite базы данных для хранения историй экспериментов в формате Optuna"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Создание таблицы trials (совместимой с Optuna форматом)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            number INTEGER,                    -- Номер trial
            state TEXT DEFAULT 'COMPLETE',     -- Состояние trial
            value REAL,                        -- Основное значение (целевая метрика)
            datetime_start TEXT,              -- Время начала
            datetime_complete TEXT,           -- Время завершения
            params TEXT,                      -- Параметры в JSON
            distributions TEXT,               -- Распределения параметров
            user_attrs TEXT,                  -- Пользовательские атрибуты
            system_attrs TEXT,                -- Системные атрибуты
            intermediate_values TEXT          -- Промежуточные значения
        )
    """)

    # Создание таблицы для хранения истории сообщений
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT
        )
    """)

    # Создание таблицы для системных атрибутов study
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_system_attrs (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()
    logging.info(f"Optuna-compatible database initialized at {db_path}")


init_optuna_db()


def convert_function_response_to_optuna_format(function_response):
    """
    Конвертирует результаты эксперимента в формат Optuna
    """
    try:
        # Извлекаем все метрики
        epochs = function_response.get("epochs", [])
        train_losses = function_response.get("train_loss", [])
        train_mae = function_response.get("train_mae", [])
        val_losses = function_response.get("val_loss", [])
        val_mae = function_response.get("val_mae", [])

        if not val_losses:
            return None

        # Находим лучшие значения
        best_val_loss = float(min(val_losses))
        best_epoch_idx = val_losses.index(best_val_loss)

        # Формируем user_attrs (дополнительные метрики)
        user_attrs = {
            "best_val_loss": best_val_loss,
            "best_epoch": int(epochs[best_epoch_idx])
            if epochs and len(epochs) > best_epoch_idx
            else best_epoch_idx,
            "final_val_loss": float(val_losses[-1]),
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "best_val_mae": float(val_mae[best_epoch_idx])
            if val_mae and len(val_mae) > best_epoch_idx
            else None,
            "final_val_mae": float(val_mae[-1]) if val_mae else None,
            "best_train_loss": float(train_losses[best_epoch_idx])
            if train_losses and len(train_losses) > best_epoch_idx
            else None,
            "best_train_mae": float(train_mae[best_epoch_idx])
            if train_mae and len(train_mae) > best_epoch_idx
            else None,
        }

        # Рассчитываем дополнительные метрики
        if train_losses and len(train_losses) > best_epoch_idx:
            user_attrs["overfitting"] = float(
                train_losses[best_epoch_idx] - best_val_loss
            )

        # Добавляем полную историю метрик
        user_attrs["full_history"] = {
            "epochs": [int(x) for x in epochs] if epochs else [],
            "train_loss": [float(x) for x in train_losses] if train_losses else [],
            "val_loss": [float(x) for x in val_losses] if val_losses else [],
            "train_mae": [float(x) for x in train_mae] if train_mae else [],
            "val_mae": [float(x) for x in val_mae] if val_mae else [],
        }

        return {"value": best_val_loss, "user_attrs": user_attrs}

    except Exception as e:
        logging.error(f"Error converting function response to Optuna format: {e}")
        return None


def log_trial_to_optuna(
    trial_number, params, function_response, db_path="optuna_llm_logs.db"
):
    """
    Логирование trial в формате Optuna
    """
    try:
        # Конвертируем результаты в формат Optuna
        optuna_format = convert_function_response_to_optuna_format(function_response)

        if optuna_format is None:
            logging.warning(
                f"Could not convert function response for trial {trial_number}"
            )
            return None

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Проверяем, существует ли таблица с правильной структурой
        cursor.execute("PRAGMA table_info(trials)")
        columns = [column[1] for column in cursor.fetchall()]

        # Если таблица не соответствует ожидаемой структуре, пересоздаем её
        if "number" not in columns or "value" not in columns:
            logging.info("Recreating trials table with correct structure")
            cursor.execute("DROP TABLE IF EXISTS trials")
            cursor.execute("""
                CREATE TABLE trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    number INTEGER,
                    state TEXT DEFAULT 'COMPLETE',
                    value REAL,
                    datetime_start TEXT,
                    datetime_complete TEXT,
                    params TEXT,
                    distributions TEXT,
                    user_attrs TEXT,
                    system_attrs TEXT,
                    intermediate_values TEXT
                )
            """)

        # Вставляем trial в формате Optuna
        cursor.execute(
            """
            INSERT INTO trials (
                number, state, value, datetime_start, datetime_complete,
                params, distributions, user_attrs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_number,
                "COMPLETE",
                optuna_format["value"],
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(params, ensure_ascii=False),
                json.dumps(
                    {}, ensure_ascii=False
                ),  # distributions - пустой для фиксированных параметров
                json.dumps(optuna_format["user_attrs"], ensure_ascii=False),
            ),
        )

        conn.commit()
        conn.close()

        logging.info(
            f"Trial logged to Optuna DB: trial_number={trial_number}, "
            f"value={optuna_format['value']}"
        )

        return optuna_format["value"]

    except Exception as e:
        logging.error(f"Error logging trial to Optuna: {e}")
        return None


def log_message_to_history(role, content, db_path="optuna_llm_logs.db"):
    """Логирование сообщений в историю разговора"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversation_history (timestamp, role, content)
            VALUES (?, ?, ?)
        """,
            (datetime.now().isoformat(), role, content),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error logging message to history: {e}")


def get_study_summary(db_path="optuna_llm_logs.db"):
    """Получение сводки по study в формате Optuna"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Проверяем существование таблицы
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trials'"
        )
        if not cursor.fetchone():
            conn.close()
            return None

        # Получаем общее количество trials
        cursor.execute("SELECT COUNT(*) FROM trials WHERE value IS NOT NULL")
        n_trials_result = cursor.fetchone()
        n_trials = n_trials_result[0] if n_trials_result else 0

        # Получаем лучший trial
        cursor.execute(
            "SELECT * FROM trials WHERE value IS NOT NULL ORDER BY value ASC LIMIT 1"
        )
        best_trial = cursor.fetchone()

        # Получаем все trials для статистики
        cursor.execute("SELECT value FROM trials WHERE value IS NOT NULL")
        values_result = cursor.fetchall()
        values = [row[0] for row in values_result] if values_result else []

        conn.close()

        summary = {
            "n_trials": n_trials,
            "best_trial": best_trial,
            "best_value": min(values) if values else None,
            "values": values,
        }

        return summary

    except Exception as e:
        logging.error(f"Error getting study summary: {e}")
        return None


# --- Остальная часть кода остается без изменений ---

load_dotenv()

provider = "openrouter"
ITERATIONS = 10

try:
    if provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        # model = "alibaba/tongyi-deepresearch-30b-a3b:free"
        model = "google/gemini-3-pro-preview"
    elif provider == "local":
        client = OpenAI(
            base_url="http://0.0.0.0:30000/v1",
            api_key="None",
        )

        model = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    logging.info(f"Using model: {model} by {provider}")
    print(f"Using model: {model} by {provider}")

except Exception as e:
    logging.error(f"Error during setup: {e}")
    raise

# --- 2. DEFINE THE TOOL (SCHEMA AND IMPLEMENTATION) ---
available_functions = {
    "run_informer_experiment": run_informer_experiment,
}

# --- 3. THE OPTIMIZATION LOOP ---
messages = [
    {
        "role": "user",
        "content": "Your goal is to find the best hyperparameter values for the Informer model to minimize MSE on the ETDataset (Electricity Transformer Dataset) for time-series forecasting task.\n**DATASET DETAILS** The dataset columns include information on the date (recorded date), HUFL (High UseFul Load), HULL (High UseLess Load), MUFL (Middle UseFul Load), MULL (Middle UseLess Load), LUFL (Low UseFul Load), LULL (Low UseLess Load), and a target column OT (Oil Temperature). "
        "You have a function that runs a long sequence time-series forecasting experiment using the Informer model. "
        "Start by suggesting initial values for x and y to test. Do not stop suggesting new values. Here's the range of hyperparameters for reference, but you can use any values in between these ranges: seq_len_choices: Sequence[int] = (48, 96, 168, 336, 720); label_len_choices: Sequence[int] = (24, 48, 96, 168, 336); pred_len_choices: Sequence[int] = (24, 48, 96, 168, 336, 720); d_model_choices: Sequence[int] = (256, 512); n_heads_choices: Sequence[int] = (4, 8); e_layers_choices: Sequence[int] = (2, 3); d_layers_choices: Sequence[int] = (1, 2); d_ff_choices: Sequence[int] = (512, 1024, 2048); factor_choices: Sequence[int] = (1, 3, 5); dropout: tuple[float, float] = (0.01, 0.3); learning_rate: tuple[float, float] = (1e-5, 5e-4); batch_size_choices: Sequence[int] = (16, 32, 64); train_epochs: tuple[int, int] = (4, 20); patience: tuple[int, int] = (2, 6); s_layers_choices: Sequence[str] = ('3,2,1', '4,3,2'); attn_choices: Sequence[str] = ('prob', 'full'); embed_choices: Sequence[str] = ('timeF', 'fixed', 'learned'); activation_choices: Sequence[str] = ('gelu', 'relu'); distil_options: Sequence[bool] = (True, False); output_attention_options: Sequence[bool] = (False, True); mix_options: Sequence[bool] = (True, False); padding_options: Sequence[int] = (0, 1); lradj_choices: Sequence[str] = ('type1', 'type2')."
        f"Use both exploration and exploitation methods. You have {ITERATIONS} iterations. ",
    }
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

        messages.append(response_message)
        log_message_to_history("assistant", str(response_message))

        if not response_message.tool_calls:
            logging.info("LLM finished or decided not to call a tool. Stopping loop.")
            print("LLM finished or decided not to call a tool. Stopping loop.")
            break

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
            print(f"Results: {function_response}")

            # Логирование в формате Optuna
            trial_value = log_trial_to_optuna(
                trial_number=i + 1,
                params=function_args,
                function_response=function_response,
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
        logging.error(f"JSON decode error on iteration {i + 1}: {e}")
        print(f"JSON decode error: {e}")
    except Exception as e:
        logging.error(f"Error during iteration {i + 1}: {e}")
        print(f"Error during iteration {i + 1}: {e}")

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
    messages.append({"role": "user", "content": final_prompt})

    log_message_to_history("user", final_prompt)

    final_summary_response = client.chat.completions.create(
        model=model, messages=messages
    )

    logging.info(
        f"Final summary from LLM: {final_summary_response.choices[0].message.content}"
    )
    print("\n✅ Final Report from the Agent:")
    print(final_summary_response.choices[0].message.content)
    log_message_to_history(
        "assistant", final_summary_response.choices[0].message.content
    )
except Exception as e:
    logging.error(f"Error during final summary request: {e}")
    print(f"Error during final summary request: {e}")


# --- 5. OPTUNA DASHBOARD GENERATION ---
def create_optuna_dashboard():
    """Создание визуализаций в формате Optuna Dashboard"""
    print("\n ====== Creating Optuna Dashboard ======")

    try:
        # Загрузка данных с проверкой структуры таблицы
        conn = sqlite3.connect("optuna_llm_logs.db")

        # Проверяем существование таблицы и её структуру
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trials'"
        )
        if not cursor.fetchone():
            print("No trials table found in database")
            conn.close()
            return

        # Получаем информацию о колонках
        cursor.execute("PRAGMA table_info(trials)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Available columns: {columns}")

        # Проверяем обязательные колонки
        required_columns = ["trial_id", "value"]
        missing_columns = [col for col in required_columns if col not in columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            # Создаем таблицу заново если она неправильная
            cursor.execute("DROP TABLE IF EXISTS trials")
            cursor.execute("""
                CREATE TABLE trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    number INTEGER,
                    state TEXT DEFAULT 'COMPLETE',
                    value REAL,
                    datetime_start TEXT,
                    datetime_complete TEXT,
                    params TEXT,
                    distributions TEXT,
                    user_attrs TEXT,
                    system_attrs TEXT,
                    intermediate_values TEXT
                )
            """)
            conn.commit()
            print("Recreated trials table with correct structure")
            conn.close()
            return

        # Загружаем данные
        query = """
            SELECT trial_id, number, value, params, user_attrs, datetime_start 
            FROM trials 
            WHERE value IS NOT NULL 
            ORDER BY value ASC
        """

        try:
            trials_df = pd.read_sql_query(query, conn)
        except Exception as query_error:
            print(f"Query error: {query_error}")
            # Альтернативный запрос если колонка number отсутствует
            try:
                trials_df = pd.read_sql_query(
                    """
                    SELECT trial_id, value, params, user_attrs, datetime_start 
                    FROM trials 
                    WHERE value IS NOT NULL 
                    ORDER BY value ASC
                """,
                    conn,
                )
                trials_df["number"] = trials_df[
                    "trial_id"
                ]  # Используем trial_id как номер
            except Exception as alt_error:
                print(f"Alternative query also failed: {alt_error}")
                conn.close()
                return

        conn.close()

        if trials_df.empty:
            print("No trials data available for dashboard")
            return

        print(f"Total trials analyzed: {len(trials_df)}")

        # Создание комплексной визуализации в стиле Optuna
        fig = plt.figure(figsize=(20, 15))

        # 1. Optimization History (основной график Optuna)
        plt.subplot(3, 3, 1)
        if "number" in trials_df.columns:
            trials_df_sorted = trials_df.sort_values("number")
            plt.plot(
                trials_df_sorted["number"], trials_df_sorted["value"], "o-", alpha=0.7
            )
        else:
            plt.plot(range(len(trials_df)), trials_df["value"], "o-", alpha=0.7)

        plt.axhline(
            y=trials_df["value"].min(),
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Best: {trials_df['value'].min():.4f}",
        )
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value (Loss)")
        plt.title("Optimization History")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Параллельные координаты для параметров (упрощенная версия)
        plt.subplot(3, 3, 2)
        # Извлекаем параметры x и y
        x_params = []
        y_params = []
        values = []

        for _, row in trials_df.iterrows():
            try:
                params = (
                    json.loads(row["params"])
                    if isinstance(row["params"], str)
                    else row["params"]
                )
                if params and "x" in params and "y" in params:
                    x_params.append(params["x"])
                    y_params.append(params["y"])
                    values.append(row["value"])
            except Exception:
                continue

        if x_params and y_params:
            scatter = plt.scatter(
                x_params, y_params, c=values, cmap="viridis", alpha=0.7
            )
            plt.colorbar(scatter)
            plt.xlabel("x parameter")
            plt.ylabel("y parameter")
            plt.title("Parameter Space")

        # 3. Distribution of objective values
        plt.subplot(3, 3, 3)
        plt.hist(trials_df["value"], bins=20, alpha=0.7, edgecolor="black")
        plt.axvline(
            trials_df["value"].mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {trials_df['value'].mean():.4f}",
        )
        plt.axvline(
            trials_df["value"].min(),
            color="g",
            linestyle="--",
            label=f"Min: {trials_df['value'].min():.4f}",
        )
        plt.xlabel("Objective Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Objective Values")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Timeline of trials
        plt.subplot(3, 3, 4)
        plt.plot(range(len(trials_df)), trials_df["value"].cummin(), "g-", linewidth=2)
        plt.xlabel("Trial Order")
        plt.ylabel("Best Value So Far")
        plt.title("Best Value Progress")
        plt.grid(True, alpha=0.3)

        # 5. Scatter plot of all trials colored by value
        plt.subplot(3, 3, 5)
        trial_numbers = (
            range(len(trials_df))
            if "number" not in trials_df.columns
            else trials_df["number"]
        )
        objective_values = trials_df["value"]
        scatter = plt.scatter(
            trial_numbers,
            objective_values,
            c=objective_values,
            cmap="plasma",
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter)
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value")
        plt.title("All Trials Scatter Plot")
        plt.grid(True, alpha=0.3)

        # 6. Best parameters analysis (if any)
        plt.subplot(3, 3, 6)
        if len(trials_df) > 0:
            best_trial_idx = trials_df["value"].idxmin()
            best_trial = trials_df.loc[best_trial_idx]
            try:
                best_params = (
                    json.loads(best_trial["params"])
                    if isinstance(best_trial["params"], str)
                    else best_trial["params"]
                )
                if best_params:
                    param_names = list(best_params.keys())[
                        :8
                    ]  # Ограничиваем для читаемости
                    param_values = [
                        best_params[name] for name in param_names if name in best_params
                    ]

                    if param_values:
                        bars = plt.bar(range(len(param_names)), param_values)
                        plt.xlabel("Parameters")
                        plt.ylabel("Values")
                        plt.title("Best Trial Parameters")
                        plt.xticks(range(len(param_names)), param_names, rotation=45)
                        plt.grid(True, alpha=0.3)
                    else:
                        plt.text(
                            0.5,
                            0.5,
                            "No parameters\navailable",
                            ha="center",
                            va="center",
                            transform=plt.gca().transAxes,
                        )
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "No parameters\navailable",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
            except Exception:
                plt.text(
                    0.5,
                    0.5,
                    "Parameter analysis\nnot available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
        else:
            plt.text(
                0.5,
                0.5,
                "No trials data\navailable",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        # 7. Convergence plot
        plt.subplot(3, 3, 7)
        cumulative_best = []
        sorted_values = sorted(trials_df["value"])
        for i in range(1, len(sorted_values) + 1):
            cumulative_best.append(min(sorted_values[:i]))

        plt.plot(range(1, len(cumulative_best) + 1), cumulative_best, "b-", linewidth=2)
        plt.xlabel("Number of Trials")
        plt.ylabel("Best Value")
        plt.title("Convergence Plot")
        plt.grid(True, alpha=0.3)

        # 8. User attributes analysis (optional)
        plt.subplot(3, 3, 8)
        overfitting_values = []

        for _, row in trials_df.iterrows():
            try:
                user_attrs = (
                    json.loads(row["user_attrs"])
                    if isinstance(row["user_attrs"], str)
                    else row["user_attrs"]
                )
                if (
                    user_attrs
                    and "overfitting" in user_attrs
                    and user_attrs["overfitting"] is not None
                ):
                    overfitting_values.append(user_attrs["overfitting"])
            except Exception:
                continue

        if overfitting_values:
            plt.hist(overfitting_values, bins=15, alpha=0.7, edgecolor="black")
            plt.xlabel("Overfitting (Train - Val Loss)")
            plt.ylabel("Frequency")
            plt.title("Overfitting Distribution")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(
                0.5,
                0.5,
                "Overfitting analysis\nnot available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        plt.axis("off")
        stats_text = f"""
        OPTUNA DASHBOARD SUMMARY
        
        Total Trials: {len(trials_df)}
        Best Value: {trials_df["value"].min():.6f}
        Mean Value: {trials_df["value"].mean():.4f}
        Std Value: {trials_df["value"].std():.4f}
        Worst Value: {trials_df["value"].max():.4f}
        
        Database: optuna_llm_logs.db
        Compatible with Optuna Dashboard
        """
        plt.text(
            0.1,
            0.5,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout()
        plt.savefig("optuna_dashboard.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("✅ Optuna dashboard saved as 'optuna_dashboard.png'")

        # CSV report
        report_data = []
        for _, row in trials_df.iterrows():
            try:
                params = (
                    json.loads(row["params"])
                    if isinstance(row["params"], str)
                    else row["params"]
                )
                user_attrs = (
                    json.loads(row["user_attrs"])
                    if isinstance(row["user_attrs"], str)
                    else row["user_attrs"]
                )

                report_row = {
                    "trial_number": row.get("number", row["trial_id"]),
                    "value": row["value"],
                    "datetime": row.get("datetime_start", ""),
                }

                # Добавляем параметры
                if params:
                    report_row.update({f"param_{k}": v for k, v in params.items()})

                # Добавляем пользовательские атрибуты (кроме сложных структур)
                if user_attrs:
                    simple_attrs = {
                        f"attr_{k}": v
                        for k, v in user_attrs.items()
                        if not isinstance(v, (dict, list))
                    }
                    report_row.update(simple_attrs)

                report_data.append(report_row)
            except Exception:
                continue

        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.to_csv("optuna_trials_report.csv", index=False)
            print("✅ Detailed report saved as 'optuna_trials_report.csv'")

    except ImportError as e:
        print(f"Analysis libraries not available: {e}")
        print("Install required packages: pip install matplotlib pandas numpy")
    except Exception as e:
        print(f"Error during dashboard creation: {e}")
        logging.error(f"Error during dashboard creation: {e}")


# --- 6. EXECUTE DASHBOARD CREATION ---
try:
    create_optuna_dashboard()

    # Вывод финальной сводки
    summary = get_study_summary()
    if summary:
        print("\n===== OPTUNA STUDY SUMMARY =====")
        print(f"Total trials: {summary['n_trials']}")
        print(f"Best value: {summary['best_value']}")
        if summary["values"]:
            print(f"Mean value: {np.mean(summary['values']):.4f}")
            print(f"Std value: {np.std(summary['values']):.4f}")

except Exception as e:
    logging.error(f"Error during dashboard creation: {e}")
    print(f"Error during dashboard creation: {e}")

print("\n📊 To use with Optuna Dashboard:")
print(
    "Run: optuna-dashboard --study-name informer_optimization --storage sqlite:///optuna_llm_logs.db"
)
