import subprocess
import time

script = ["python", "main.py", "--download"]
# keep running the script until it executes successfully
while True:
    print(f"Starting {script}")
    process = subprocess.Popen(script)
    process.wait()

    if process.returncode == 0:
        # Exit the loop if the script executed successfully
        break
    else:
        print(f"Error detected. Restarting {script}...")
        time.sleep(1)
