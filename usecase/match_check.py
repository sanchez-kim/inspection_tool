from pathlib import Path
from natsort import natsorted
import re

obj_path = natsorted(Path("../output/objs").glob("*.png"))
img_path = natsorted(Path("../output/images").glob("*.png"))


missing = []

for obj in obj_path:
    if obj.name not in [img.name for img in img_path]:
        print(obj.stem)
        missing.append(obj.stem)

print(missing)

# img_num = []

# for item in img_path:
#     match = re.match(r"M(\d+)_S(\d+)_F(\d+)", item.stem)
#     if match:
#         img_num.append(int(match.group(2)))

# img_num = sorted(list(set(img_num)))
# for i in range(1, 5000):
#     if i not in img_num:
#         print(i)
