import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import optuna
from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

# Optimization settings
ITERATIONS = 10
API_DELAY = 2  # Seconds to wait between API calls to prevent rate limiting

# API Configuration
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Models to evaluate
MODELS_TO_COMPARE = [
    "qwen/qwen3-8b",
    "google/gemini-2.0-flash-001", # Updated to a current stable model example
]

# Initialize Client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print("--- SYSTEM CONFIGURATION ---")
print(f"Provider URL:   {BASE_URL}")
print(f"Iterations:     {ITERATIONS}")
print(f"API Delay:      {API_DELAY} seconds")
print(f"Models:         {MODELS_TO_COMPARE}")
print("-" * 30 + "\n")


# --- 2. OBJECTIVE FUNCTION ---

def calculate_loss(x: float, y: float) -> float:
    """
    The target black-box function to minimize.
    Global minimum is at (3, 5) with a value of 0.
    """
    return (x - 3) ** 2 + (y - 5) ** 2


# --- 3. LLM OPTIMIZATION AGENT ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_loss",
            "description": "Calculates the loss for given parameters. Goal: MINIMIZE loss towards 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Float value between -10 and 10"},
                    "y": {"type": "number", "description": "Float value between -10 and 10"}
                },
                "required": ["x", "y"]
            }
        }
    }
]

def run_llm_optimization(model_name, trial_count):
    """
    Runs an optimization loop using an LLM agent.
    """
    print(f"{'='*60}")
    print(f"[INFO] Starting Agent: {model_name}")
    print(f"{'='*60}")
    
    system_prompt = (
        f"You are an optimization algorithm. Your goal is to minimize a black-box function. "
        f"You have {trial_count} allowed attempts. "
        "Search Space: x in [-10, 10], y in [-10, 10]. "
        "Analyze previous results to inform your next guess. "
        "You MUST call the function 'get_loss' with arguments 'x' and 'y'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Begin optimization."}
    ]

    trajectory = []
    best_loss = float('inf')
    best_params = {}

    for i in range(trial_count):
        print(f"\n--- Step {i+1}/{trial_count} ---")
        
        try:
            # API Call
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2 # Low temperature for more deterministic reasoning
            )
            duration = time.time() - start_time
            
            msg = response.choices[0].message
            messages.append(msg)

            # Handle Tool Calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    
                    try:
                        args = json.loads(args_str)
                        x = float(args.get('x', 0))
                        y = float(args.get('y', 0))
                    except (json.JSONDecodeError, ValueError):
                        print(f"  [ERROR] Failed to parse arguments: {args_str}")
                        continue

                    # Execute Function
                    loss = calculate_loss(x, y)
                    print(f"  [ACTION] Suggesting parameters: x={x:.4f}, y={y:.4f}")
                    print(f"  [RESULT] Loss: {loss:.6f} (Lat: {duration:.2f}s)")
                    
                    # Track Progress
                    trajectory.append({'x': x, 'y': y, 'loss': loss, 'step': i})

                    if loss < best_loss:
                        print(f"  [UPDATE] New best found! ({best_loss:.6f} -> {loss:.6f})")
                        best_loss = loss
                        best_params = {'x': x, 'y': y}
                    else:
                        diff = loss - best_loss
                        print(f"  [INFO] No improvement (+{diff:.6f} from best)")

                    # Feed result back to LLM
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": json.dumps({"loss": loss})
                    })
            else:
                print("  [WARNING] Model did not call the tool. Reinforcing instructions.")
                messages.append({
                    "role": "user", 
                    "content": "Invalid response. You must call the 'get_loss' function to proceed."
                })

        except Exception as e:
            print(f"  [CRITICAL] API or Runtime Error: {e}")
            break
        
        # Rate Limit Handling
        if i < trial_count - 1 and API_DELAY > 0:
            time.sleep(API_DELAY)

    print(f"\n[DONE] Finished {model_name}. Best Loss: {best_loss:.6f}")
    return {
        "agent": model_name,
        "best_loss": best_loss,
        "best_params": best_params,
        "trajectory": trajectory
    }


# --- 4. OPTUNA OPTIMIZATION ENGINE ---

def run_optuna_optimization(trial_count):
    """
    Runs standard Bayesian optimization using Optuna as a baseline.
    """
    print(f"\n{'='*60}")
    print("[INFO] Starting Agent: Optuna (TPE Sampler)")
    print(f"{'='*60}")
    
    # Suppress Optuna's default verbose logging
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    trajectory = []

    def objective(trial):
        step_num = trial.number
        
        # Suggest parameters
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        
        loss = calculate_loss(x, y)
        
        # Log every few steps or just store data
        trajectory.append({'x': x, 'y': y, 'loss': loss, 'step': step_num})
        return loss

    print(f"  [INFO] Running {trial_count} trials...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trial_count)

    print(f"[DONE] Finished Optuna. Best Loss: {study.best_value:.6f}")

    return {
        "agent": "Optuna (TPE)",
        "best_loss": study.best_value,
        "best_params": study.best_params,
        "trajectory": trajectory
    }


# --- 5. VISUALIZATION ---

def plot_trajectories(results):
    """
    Generates a contour plot comparing the trajectories of different agents.
    """
    print("\n[INFO] Generating visualization...")
    
    # Setup grid for contour background
    x_range = np.linspace(-12, 12, 100)
    y_range = np.linspace(-12, 12, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 3) ** 2 + (Y - 5) ** 2 

    num_agents = len(results)
    fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 6))
    
    # Ensure axes is iterable even if there is only one plot
    if num_agents == 1:
        axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        # Clean model name for display
        name = result['agent'].split('/')[-1]
        traj = result['trajectory']
        
        # Draw Contour
        cp = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        
        # Mark Target
        ax.plot(3, 5, 'w*', markersize=18, markeredgecolor='k', label='Global Min (3,5)')

        # Extract points
        xs = [step['x'] for step in traj]
        ys = [step['y'] for step in traj]

        # Draw Path
        ax.plot(xs, ys, color='white', alpha=0.4, linewidth=1.5, linestyle='--')
        ax.plot(xs, ys, color='red', alpha=0.3, linewidth=1) 

        # Draw Steps
        for i, (x, y) in enumerate(zip(xs, ys)):
            # Color coding: Start=Green, End=Red, Middle=Orange
            color = 'orange'
            if i == 0: color = 'lime'
            elif i == len(xs)-1: color = 'red'
            
            ax.scatter(x, y, color=color, s=50, edgecolors='black', zorder=5)
            # Annotate step number
            ax.text(x + 0.4, y + 0.4, str(i+1), color='white', fontsize=9, fontweight='bold', zorder=6)

        ax.set_title(f"{name}\nMin Loss: {result['best_loss']:.4f}", fontsize=11)
        ax.set_xlabel("Parameter X")
        
        if idx == 0:
            ax.set_ylabel("Parameter Y")
            
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cp, cax=cbar_ax, label='Loss Value (Lower is Better)')

    plt.suptitle("Optimization Trajectory: LLM Agents vs Optuna", fontsize=14)
    plt.subplots_adjust(right=0.9, wspace=0.3)
    
    print("[INFO] Displaying plot window.")
    plt.show()


# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    all_results = []

    # 1. Run LLM Agents
    for model in MODELS_TO_COMPARE:
        result = run_llm_optimization(model, ITERATIONS)
        all_results.append(result)
    
    # 2. Run Optuna Baseline
    optuna_result = run_optuna_optimization(ITERATIONS)
    all_results.append(optuna_result)

    # 3. Print Summary Table
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    table_data = []
    for r in all_results:
        # Format model name for table
        agent_name = r['agent'].split('/')[-1][:25]
        best_loss = f"{r['best_loss']:.6f}"
        best_coords = f"x={r['best_params']['x']:.3f}, y={r['best_params']['y']:.3f}"
        table_data.append([agent_name, best_loss, best_coords])
    
    print(tabulate(table_data, headers=["Agent", "Best Loss", "Best Parameters"], tablefmt="heavy_grid"))

    # 4. Visual Comparison
    plot_trajectories(all_results)