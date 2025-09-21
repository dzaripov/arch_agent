import asyncio
import aiohttp
import time
import argparse

# 1. Define the API endpoint with a placeholder for the host
# MODIFIED: Added {host} placeholder
API_URL = "http://{host}:{port}/v1/chat/completions"
SIMPLE_PAYLOAD = {
    "model": "Qwen/Qwen3-4B-Thinking-2507-FP8",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 100."},
    ],
    "max_tokens": 100,
    "temperature": 0.1,
}

# 2. An asynchronous function to send one request (no changes here)
async def send_request(session, url, semaphore, request_id):
    """Acquires semaphore, sends a request, and prints the outcome."""
    async with semaphore:
        try:
            async with session.post(url, json=SIMPLE_PAYLOAD) as response:
                if response.status == 200:
                    print(f"Request {request_id}: Success (Status {response.status})")
                else:
                    text = await response.text()
                    print(f"Request {request_id}: Error (Status {response.status}) - {text[:100]}")
        except Exception as e:
            print(f"Request {request_id}: Failed with exception: {e}")

# 3. Main function to orchestrate the test
async def main(args):
    """Sets up and runs the concurrent API requests."""
    start_time = time.time()
    # MODIFIED: Format the URL with both the host and port from args
    url = API_URL.format(host=args.host, port=args.port)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    print(f"Starting stability test on {url} with {args.num_requests} requests and {args.concurrency} concurrency level...")

    async with aiohttp.ClientSession() as session:
        for i in range(args.num_requests):
            task = asyncio.create_task(send_request(session, url, semaphore, i + 1))
            tasks.append(task)
        await asyncio.gather(*tasks)

    duration = time.time() - start_time
    print("\n--- Test Complete ---")
    print(f"Sent {args.num_requests} requests in {duration:.2f} seconds.")
    print(f"Requests per second: {args.num_requests / duration:.2f}")

# 4. Command-line argument parsing and script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple API stability testing script.")
    # ADDED: New argument for the host/IP address
    parser.add_argument("--host", type=str, default="localhost", help="The IP address or hostname of the server.")
    parser.add_argument("--num-requests", type=int, default=100, help="Total number of requests to send.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of requests to send concurrently.")
    parser.add_argument("--port", type=int, default=30000, help="The port your API server is running on.")
    cli_args = parser.parse_args()
    asyncio.run(main(cli_args))