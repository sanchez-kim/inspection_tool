import shutil
from tqdm import tqdm

with open("../statistics/M_pass_fail.txt", "r") as f:
    data = f.read().splitlines()

filelist = ["../output/objs/" + item + ".png" for item in data]

for file in tqdm(filelist):
    shutil.copy(file, "../retry/objs")
