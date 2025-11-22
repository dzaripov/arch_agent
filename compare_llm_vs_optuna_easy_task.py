import os
import json
import optuna
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate
import time

# --- 1. SETUP ---
load_dotenv()

ITERATIONS = 10 
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Define models to compare
MODELS_TO_COMPARE = [
    "qwen/qwen3-8b",
    "google/gemini-3-pro-preview",
]

print("--- SYSTEM CONFIGURATION ---")
print(f"Provider URL: {BASE_URL}")
print(f"Iterations per agent: {ITERATIONS}")
print("Sleep per API call: 15 seconds")
print(f"Models: {MODELS_TO_COMPARE}")
print("----------------------------\n")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- 2. THE OBJECTIVE FUNCTION ---

def calculate_loss(x: float, y: float) -> float:
    """Target function: f(x, y) = (x - 3)^2 + (y - 5)^2"""
    val = (x - 3) ** 2 + (y - 5) ** 2
    return val

# --- 3. LLM OPTIMIZATION ENGINE ---

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_loss",
            "description": "Calculates loss. Goal: MINIMIZE loss. Optimal is 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Float [-10, 10]"},
                    "y": {"type": "number", "description": "Float [-10, 10]"}
                },
                "required": ["x", "y"]
            }
        }
    }
]

def run_llm_optimization(model_name, trial_count):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING AGENT: {model_name}")
    print(f"{'='*60}")
    
    messages = [
        {"role": "system", "content": (
            f"Minimize the black-box function. You have {trial_count} steps. "
            "Range: [-10, 10]. Learn from previous steps. "
            "Output arguments for function 'get_loss'."
        )},
        {"role": "user", "content": "Start optimization."}
    ]

    trajectory = [] 
    best_loss = float('inf')
    best_params = {}

    def tool_get_loss(x, y):
        return calculate_loss(x, y)

    available_functions = {"get_loss": tool_get_loss}

    for i in range(trial_count):
        print(f"\n--- Step {i+1}/{trial_count} ---")
        
        try:
            print("  > Sending prompt to LLM...")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model_name, messages=messages, tools=tools_schema, 
                tool_choice="auto", temperature=0.2
            )
            
            duration = time.time() - start_time
            print(f"  < Received response in {duration:.2f}s")
            
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                print(f"  > LLM requested {len(msg.tool_calls)} tool(s).")
                
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    print(f"  > Parsing arguments: {args_str}")
                    
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        print("  ! ERROR: Failed to parse JSON arguments. Skipping.")
                        continue

                    x, y = args.get('x'), args.get('y')
                    
                    if x is None or y is None: 
                        print("  ! WARNING: LLM provided None for x or y. Defaulting to 0.")
                        x, y = 0, 0

                    print(f"  > Executing '{func_name}' with x={x}, y={y}...")
                    loss = available_functions["get_loss"](x, y)
                    print(f"  < Result: Loss = {loss:.6f}")
                    
                    trajectory.append({'x': x, 'y': y, 'loss': loss, 'step': i})

                    if loss < best_loss:
                        print(f"  *** NEW BEST FOUND! (Previous: {best_loss:.6f} -> New: {loss:.6f}) ***")
                        best_loss = loss
                        best_params = {'x': x, 'y': y}
                    else:
                        diff = loss - best_loss
                        print(f"  (Not an improvement. +{diff:.6f} from best)")

                    messages.append({
                        "tool_call_id": tool_call.id, "role": "tool", 
                        "name": func_name, 
                        "content": json.dumps({"loss": loss})
                    })
            else:
                print("  ! WARNING: LLM did not call a tool.")
                content = msg.content
                print(f"  > LLM Message: {content[:100]}...") 
                messages.append({"role": "user", "content": "You must call the get_loss tool."})

        except Exception as e:
            print(f"  !!! CRITICAL ERROR: {e}")
            break
        
        # --- SLEEP BLOCK ---
        # We sleep after every iteration (except the last one) to avoid Rate Limits
        # if i < trial_count - 1:
        #     print("  > 💤 Sleeping 15s to avoid Rate Limits...")
        #     time.sleep(15)
        # -------------------

    print(f"\n🏁 Finished {model_name}. Best Loss: {best_loss}")
    return {"agent": model_name, "best_loss": best_loss, "best_params": best_params, "trajectory": trajectory}

# --- 4. OPTUNA ENGINE ---

def run_optuna_optimization(trial_count):
    print(f"\n{'='*60}")
    print("🚀 STARTING AGENT: Optuna (TPE Sampler)")
    print(f"{'='*60}")
    
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    trajectory = []

    def objective(trial):
        step_num = trial.number
        print(f"\n--- Trial {step_num+1}/{trial_count} ---")
        
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        print(f"  > Algorithm suggests: x={x:.4f}, y={y:.4f}")
        
        loss = calculate_loss(x, y)
        print(f"  < Result: Loss = {loss:.6f}")

        trajectory.append({'x': x, 'y': y, 'loss': loss, 'step': step_num})
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print("  > Initializing Study...")
    study.optimize(objective, n_trials=trial_count)
    print("  > Study Finalized.")

    print(f"\n🏁 Finished Optuna. Best Loss: {study.best_value}")

    return {
        "agent": "Optuna (TPE)",
        "best_loss": study.best_value,
        "best_params": study.best_params,
        "trajectory": trajectory
    }

# --- 5. VISUALIZATION ---

def plot_trajectories(results):
    print("\n🎨 Generating Visualization Plots...")
    
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X - 3) ** 2 + (Y - 5) ** 2 

    num_agents = len(results)
    fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 6))
    if num_agents == 1: axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        name = result['agent'].split('/')[-1]
        traj = result['trajectory']
        
        print(f"  > Plotting trajectory for {name} ({len(traj)} points)...")

        cp = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        ax.plot(3, 5, 'w*', markersize=18, markeredgecolor='k', label='Target (3,5)')

        xs = [step['x'] for step in traj]
        ys = [step['y'] for step in traj]

        ax.plot(xs, ys, color='white', alpha=0.4, linewidth=1.5, linestyle='--')
        ax.plot(xs, ys, color='red', alpha=0.3, linewidth=1) 

        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == 0: c = 'lime'
            elif i == len(xs)-1: c = 'red'
            else: c = 'orange'
            
            ax.scatter(x, y, color=c, s=50, edgecolors='black', zorder=5)
            ax.text(x + 0.4, y + 0.4, str(i+1), color='white', fontsize=10, fontweight='bold', zorder=6)

        ax.set_title(f"{name}\nMin Loss: {result['best_loss']:.4f}", fontsize=11)
        ax.set_xlabel("X Parameter")
        if idx == 0: ax.set_ylabel("Y Parameter")
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cp, cax=cbar_ax, label='Loss Value (Lower is Better)')

    plt.suptitle("Optimization Race: LLMs vs Optuna (Target x=3, y=5)", fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.3)
    
    print("  > Showing plot window...")
    plt.show()

# --- 6. MAIN ---

if __name__ == "__main__":
    all_results = []

    # Run LLMs
    for m in MODELS_TO_COMPARE:
        res = run_llm_optimization(m, ITERATIONS)
        all_results.append(res)
    
    # Run Optuna (No sleep needed for Optuna)
    res_opt = run_optuna_optimization(ITERATIONS)
    all_results.append(res_opt)

    # Print Summary Table
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    
    table_data = []
    for r in all_results:
        table_data.append([
            r['agent'].split('/')[-1][:25],
            f"{r['best_loss']:.6f}", 
            f"x={r['best_params']['x']:.3f}, y={r['best_params']['y']:.3f}"
        ])
    
    print(tabulate(table_data, headers=["Agent", "Best Loss", "Best Params"], tablefmt="heavy_grid"))

    # Plot
    plot_trajectories(all_results)