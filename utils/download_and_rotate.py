import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv


def rotate_image(image, angle):
    return image.rotate(angle)


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


def process_images(file_list, base_image_url):
    for file_name in file_list:
        # Extract model_num, sentence_num, and frame_num from the file name
        parts = file_name.split("_")
        model_num = parts[0][1:]  # Remove 'M' and get model_num
        sentence_num = parts[1][1:]  # Remove 'S' and get sentence_num
        frame_num = parts[2][1:].split(".")[0]  # Remove 'F' and '.png' to get frame_num

        image_url = base_image_url.format(
            model_num=model_num, sentence_num=sentence_num, frame_num=frame_num
        )

        try:
            download_and_process_image(image_url, f"./{file_name}")
        except Exception as e:
            print(e)


load_dotenv()

base_image_url = os.getenv("base_image_url")

file_list = [
    "M07_S3461_F036.png",
    "M07_S3461_F052.png",
    "M07_S3461_F056.png",
    "M07_S3461_F089.png",
    "M07_S3463_F078.png",
    "M07_S3464_F062.png",
    "M07_S3464_F080.png",
    "M07_S3464_F081.png",
    "M07_S3468_F018.png",
    "M07_S3471_F108.png",
    "M07_S3472_F009.png",
    "M07_S3472_F080.png",
    "M07_S3475_F077.png",
    "M07_S3475_F079.png",
    "M07_S3475_F082.png",
    "M07_S3475_F083.png",
    "M07_S3475_F094.png",
    "M07_S3475_F114.png",
    "M07_S3479_F050.png",
]

process_images(file_list, base_image_url)

# for single image
# model_num = "07"
# sentence_num = "3464"
# frame_num = "078"

# image_url = base_image_url.format(
#     model_num=model_num, sentence_num=sentence_num, frame_num=frame_num
# )
# print(image_url)

# try:
#     download_and_process_image(
#         image_url, f"./M{model_num}_S{sentence_num}_F{frame_num}.png"
#     )
# except Exception as e:
#     print(e)
