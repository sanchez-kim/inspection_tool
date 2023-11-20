from pathlib import Path
from utils.filelist import extract_filenames


# finding unlisted files on json_files_list.txt
def check_unlisted_files(directory, listed_files, extension=".png"):
    unlisted_files = []
    all_files_in_directory = set(f.stem for f in Path(directory).glob(f"*{extension}"))

    for file_name in all_files_in_directory:
        if file_name not in listed_files:
            unlisted_files.append(file_name)

    return unlisted_files


def find_unlisted_files(file_list_path, directory):
    listed_file_names = set(extract_filenames(file_list_path))

    unlisted_files = check_unlisted_files(directory, listed_file_names)
    unlisted_files = sorted(item + ".png" for item in unlisted_files)
    return unlisted_files


# Check for files in 'images' directory not listed in "json_files_list.txt"
unlisted_in_images = find_unlisted_files("json_files_list.txt", "./images")

# Check for files in 'objs' directory not listed in "json_files_list.txt"
unlisted_in_objs = find_unlisted_files("json_files_list.txt", "./objs")

with open("unlisted_images.txt", "w") as f:
    for item in unlisted_in_images:
        f.write(item + "\n")

with open("unlisted_objs.txt", "w") as f:
    for item in unlisted_in_objs:
        f.write(item + "\n")
