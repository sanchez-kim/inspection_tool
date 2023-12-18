from pathlib import Path
import re
from itertools import islice
import argparse


def main(args):
    log_averages_file = args.average_path
    model_num = args.model.zfill(2)
    log_directory = f"../M{model_num}/logs"

    # Read and filter averages over 50
    averages_over_50 = []

    with open(log_averages_file, "r") as file:
        for line in islice(file, 3, None):
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

    if args.verbose:
        # Output the results
        if matched_files:
            print("List of log files with averages over 50:")
            for log_number, file_name in matched_files:
                print(f"Log {log_number}: {file_name}")
        else:
            print("No matching log files found for averages over 50.")

        print(len("Total: ", averages_over_50))

    with open(f"M{model_num}_pass_fail.txt", "w") as f:
        for log_number, file_name in matched_files:
            f.write("%s\n" % file_name.replace("_landmarks.log", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--average_path", help="Log average file path")
    parser.add_argument("-m", "--model", help="Model Number")
    parser.add_argument(
        "-v", "--verbose", help="Verbose", action="store_true", default=False
    )
    args = parser.parse_args()
    main(args)
