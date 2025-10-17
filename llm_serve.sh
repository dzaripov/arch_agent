python3 -m sglang.launch_server --model "Qwen/Qwen3-4B-Instruct-2507-FP8" --mem-fraction-static 0.85 --chunked-prefill-size -1 --max-prefill-tokens 4096 --enable-torch-compile --max-running-requests 32 --max-total-tokens 8192 --random-seed 42 --host 0.0.0.0 --port 30000 --reasoning-parser qwen3 --tool-call-parser qwen25


# python3 -m sglang.launch_server --model "Qwen/Qwen3-4B-Thinking-2507-FP8" --mem-fraction-static 0.85 --chunked-prefill-size -1 --max-prefill-tokens 1024 --enable-torch-compile --max-running-requests 32 --max-total-tokens 32768 --random-seed 42 --host 0.0.0.0 --port 30000 --reasoning-parser qwen3 --tool-call-parser qwen25

