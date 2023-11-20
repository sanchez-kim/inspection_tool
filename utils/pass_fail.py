from pathlib import Path
import re

log_averages_file = "../log_analysis_output.txt"
log_directory = "../logs"

# Read and filter averages over 50
averages_over_50 = []

with open(log_averages_file, "r") as file:
    for line in file:
        log_number = re.search(r"Log (\d+):", line).group(1)
        average = float(re.search(r"\d+\.\d+", line).group())

        if average > 50:
            averages_over_50.append(int(log_number))

# Sort the log files
log_files = sorted(Path(log_directory).glob("*.log"), key=lambda x: x.name)

# Find the filenames corresponding to the log numbers
matched_files = [
    (log_number, log_files[log_number - 1].name)
    for log_number in averages_over_50
    if (log_number - 1) < len(log_files)
]

# Output the results
if matched_files:
    print("List of log files with averages over 50:")
    for log_number, file_name in matched_files:
        print(f"Log {log_number}: {file_name}")
else:
    print("No matching log files found for averages over 50.")
