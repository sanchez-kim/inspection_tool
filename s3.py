import boto3
import os
from dotenv import load_dotenv
import re

load_dotenv()

access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("AWS_BUCKET_NAME")

obj_path = os.getenv("OBJ_PATH")
image_path = os.getenv("IMAGE_PATH")

s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)


def download_file(bucket, object_name, file_name):
    s3.download_file(bucket, object_name, file_name)
    print(f"Downloaded {object_name} to {file_name}")


# downloading failed files
with open("./statistics/M_pass_fail.txt", "r") as f:
    data = f.read().splitlines()

for item in data:
    match = re.match(r"M(\d+)_S(\d+)_F(\d+)", item)
    if match:
        model_num = match.group(1)
        sentence_num = match.group(2)
        frame_num = match.group(3)

        obj = obj_path.format(
            model_num_str=str(int(model_num)),
            model_num=model_num,
            sentence_num=sentence_num,
            frame_num=frame_num,
        )

        img = image_path.format(
            model_num_str=str(int(model_num)),
            model_num=model_num,
            sentence_num=sentence_num,
            frame_num=frame_num,
        )

        download_file(bucket_name, obj, f"./errors/{os.path.basename(obj)}")
        download_file(bucket_name, img, f"./errors/{os.path.basename(img)}")
