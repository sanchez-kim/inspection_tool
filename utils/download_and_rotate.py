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


load_dotenv()

base_image_url = os.getenv("base_image_url")
sentence_num = "2512"
frame_num = "117"

image_url = base_image_url.format(sentence_num=sentence_num, frame_num=frame_num)
print(image_url)

try:
    download_and_process_image(image_url, f"./M06_S{sentence_num}_F{frame_num}.png")
except Exception as e:
    print(e)
