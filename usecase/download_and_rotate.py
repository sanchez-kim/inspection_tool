import requests
from PIL import Image
import io
import re
from natsort import natsorted
from pathlib import Path
from dotenv import load_dotenv


def rotate_image(image, angle):
    rotated_image = image.rotate(angle, expand=True)
    new_width, new_height = image.size[1], image.size[0]
    result = rotated_image.crop((0, 0, new_width, new_height))
    return result


def download_and_process_image(image_url, image_path):
    image = fetch_image(image_url)
    if image:
        image = rotate_image(image, 90)
        image.save(image_path)


def fetch_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return None


def process_images(file_list, base_image_url, output_path):
    for file_name in file_list:
        # Extract model_num, sentence_num, and frame_num from the file name
        parts = file_name.split("_")
        model_num = parts[0][1:]  # Remove 'M' and get model_num
        sentence_num = parts[1][1:]  # Remove 'S' and get sentence_num
        frame_num = parts[2][1:].split(".")[0]  # Remove 'F' and '.png' to get frame_num

        image_url = base_image_url.format(
            model_num_str=str(int(model_num)),
            model_num=model_num,
            sentence_num=sentence_num,
            frame_num=frame_num,
        )

        try:
            download_and_process_image(image_url, f"{output_path}/{file_name}")
        except Exception as e:
            print(e)


def process_local_images(output_path):
    images = natsorted(list(Path("../temp/").glob("**/*.png")))
    for item in images:
        image = Image.open(item)
        image = rotate_image(image, 90)
        # if the file already exists, skip it
        if not Path(f"{output_path}/{item.name}").exists():
            image.save(f"{output_path}/{item.name}")


load_dotenv()

# base_image_url = os.getenv("base_image_url")
base_image_url = "https://ins-ai-speech.s3.ap-northeast-2.amazonaws.com/reprocessed_v2/원천데이터/2DImageFront/Model{model_num_str}/Sentence{sentence_num}/M{model_num}_S{sentence_num}_F{frame_num}.png"
# base_image_url = "https://ins-ai-speech.s3.ap-northeast-2.amazonaws.com/reprocessed_v2/원천데이터/2DImageFront/Model{model_num_str}/Sentence{sentence_num}/M{model_num}_S{sentence_num}_C27_F{frame_num}.png"

file_list = [
    "M04_S1838_F96.png",
]

# temp = natsorted(list(Path("../output/objs").glob("*.png")))
# for item in temp:
#     match = re.match(r"M(\d+)_S(\d+)_F(\d+)", item.stem)
#     if match:
#         sentence_num = int(match.group(2))
#         if 601 <= sentence_num <= 700 or 801 <= sentence_num <= 900:
#             file_list.append(item.name)
#         elif sentence_num in [
#             1630,
#             1680,
#             2528,
#             2529,
#             2581,
#             2593,
#             2765,
#             2773,
#             2791,
#             2792,
#             2804,
#             2821,
#             2893,
#             2897,
#             3731,
#             4561,
#         ]:
#             file_list.append(item.name)
output_path = "../errors"

# process_images(file_list, base_image_url, output_path)
process_local_images(output_path)
