
import os
import sys
import time
import signal
import subprocess
import urllib.request

SERVICES = [
    ("Brainbees_MachineLearning.src.api:app", 8000),
    ("Brainbees_MachineLearning.src.api_churn:app", 8001),
    ("Brainbees_MachineLearning.src.api_clv:app", 8002),
]

processes = []


def stop_services(*_):
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    sys.exit(0)


signal.signal(signal.SIGTERM, stop_services)
signal.signal(signal.SIGINT, stop_services)

for module, port in SERVICES:
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", module,
        "--host", "127.0.0.1", "--port", str(port)
    ])
    processes.append(process)

# Wait for each internal API to become ready.
for module, port in SERVICES:
    deadline = time.time() + 120

    while time.time() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(
                f"An internal API exited before startup: {module}"
            )

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=2
            ) as response:
                if response.status < 500:
                    break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError(f"API did not become ready on port {port}")

# Only the unified API listens on the externally accessible interface.
port = os.environ.get("PORT", "10000")
unified = subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "Brainbees_MachineLearning.src.api_all:app",
    "--host", "0.0.0.0",
    "--port", port
])
processes.append(unified)

exit_code = unified.wait()
stop_services()
