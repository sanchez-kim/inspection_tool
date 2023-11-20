from pathlib import Path

# Define the base directory and find all log files
base = "./logs"
logs = sorted(list(Path(base).glob("*.log")))

# Initialize variables for total sum and count, and a list for storing averages
total_sum = 0
total_count = 0
log_averages = []

# Process each log file
for log in logs:
    with open(log, "r") as f:
        numbers = [float(line.strip()) for line in f.readlines() if line.strip()]

    # Calculate the average for each log file if it contains numbers
    if numbers:
        log_average = sum(numbers) / len(numbers)
        log_averages.append(log_average)

        # Update the total sum and count
        total_sum += sum(numbers)
        total_count += len(numbers)

# Define the output file path
output_file_path = "log_analysis_output.txt"

# Open the output file for writing
with open(output_file_path, "w") as output_file:
    # Check if there are any numbers to calculate averages
    if total_count > 0:
        total_average = total_sum / total_count

        # Print and save the total average, total number of logs, and average per log
        print(f"Total average: {total_average}", file=output_file)
        print(f"Total number of logs: {len(logs)}", file=output_file)
        print("Average per log:", file=output_file)
        for i, avg in enumerate(log_averages, 1):
            print(f"Log {i}: {avg}", file=output_file)
    else:
        print("No data to calculate averages.", file=output_file)
