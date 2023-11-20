import os

with open("./unlisted_images.txt", "r") as f:
    unlisted_images = f.read().splitlines()

with open("./unlisted_objs.txt", "r") as f:
    unlisted_objs = f.read().splitlines()

for item in unlisted_images:
    os.remove(f"./images/{item}")

for item in unlisted_objs:
    os.remove(f"./objs/{item}")
