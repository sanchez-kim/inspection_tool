from pathlib import Path
import argparse


def main(args):
    # model_num = args.model.zfill(2)
    model_num = "retry"
    # Define the base directory and find all log files
    # base = f"./M{model_num}/logs"
    base = "retry/logs"
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
    output_file_path = f"log_analysis_output_{model_num}.txt"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model Number")
    args = parser.parse_args()
    main(args)
