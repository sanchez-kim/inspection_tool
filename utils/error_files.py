from pathlib import Path


def find_files_by_size(directory, min_size_kb, max_size_kb, output_file):
    """
    Find files in the given directory that are smaller than min_size_kb
    or larger than max_size_kb, sort them by filename, and write their paths and sizes to output_file.
    Sizes are given in kilobytes.

    :param directory: Directory to search for files.
    :param min_size_kb: Minimum file size in KB.
    :param max_size_kb: Maximum file size in KB.
    :param output_file: File to write the sorted paths and sizes of files that are out of the size range.
    """
    file_info = []

    # Gather information about files
    for file_path in directory.glob("*.png"):
        file_size_kb = file_path.stat().st_size / 1000  # Convert size to KB
        if file_size_kb < min_size_kb or file_size_kb > max_size_kb:
            file_info.append((file_path, file_size_kb))

    # Sort files by filename
    sorted_files = sorted(file_info, key=lambda x: x[0].name)

    # Write sorted information to file
    with open(output_file, "w") as file:
        for path, size in sorted_files:
            filename = str(path).replace("objs/", "").replace(".png", "")
            # file.write(f"{filename}\n")
            file.write(f"{path} - Size: {size:.2f} KB\n")


# Directory containing the images
img_output_dir = Path("./objs")

# Output file path
error_file_path = "error_files.txt"

# Find all files below 104KB or over 200KB, sort them by filename, and save their paths and sizes in error_files.txt
find_files_by_size(img_output_dir, 100, 200, error_file_path)

print("Sorted paths and sizes of files by filename saved to error_files.txt")
