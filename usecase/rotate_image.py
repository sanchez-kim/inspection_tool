from PIL import Image
from pathlib import Path

base = "./Sentence2111_images"

num = 0
for item in sorted(Path(base).glob("*.png")):
    img = Image.open(item)
    rotated = img.rotate(90, expand=True)
    output_dir = Path("./Sentence2111_images/images")
    output_dir.mkdir(exist_ok=True)
    rotated.save(f"./Sentence2111_images/images/M05_S2111_F{num:03d}.png")
    num += 1
