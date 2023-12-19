from pathlib import Path
from .filelist import extract_filenames


def check_missing_files(file_names, directory, extension=".png"):
    missing_files = []
    existing_files = set(f.stem for f in Path(directory).glob(f"*{extension}"))
    for file_name in file_names:
        if file_name not in existing_files:
            missing_files.append(file_name)
    return missing_files


def find_missing_files(file_list_path, directory):
    file_names = sorted(extract_filenames(file_list_path))
    print("Number of JSON files:", len(file_names))
    missing_files = check_missing_files(file_names, directory)
    # print("Missing files:", missing_files)
    return missing_files


# find_missing_files("json_files_list.txt", "./images")
