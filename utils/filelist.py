import re
from pathlib import Path


def extract_filenames(file_path):
    with open(file_path, "r") as file:
        file_paths = file.readlines()

    # Extract the specific part of each file path
    # Assuming the part you need always follows the format 'M06_SXXXX_FXXX'
    pattern = re.compile(r"M06_S\d{4}_F\d{3}")
    extracted_names = [
        pattern.search(Path(path).name).group(0)
        for path in file_paths
        if pattern.search(Path(path).name)
    ]

    return extracted_names
