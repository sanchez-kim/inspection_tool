import argparse
import io
import sys
import os
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import mediapipe as mp
import numpy as np
from PIL import Image
import pyrender
import requests
import trimesh
from dotenv import load_dotenv
import boto3

from utils.missing_files import find_missing_files
from models import MODELS

load_dotenv()

MAX_FRAME_NUM = 300
RESIZE_FACTOR = (1000, 1000)
BATCH = 1

base_url = os.getenv("base_url")
base_image_url = os.getenv("base_image_url")
bucket_name = "ins-ai-speech"

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def list_2dimages(bucket_name, model_num, sentence_num):
    """List frames for a given model and sentence."""
    sentence_num = str(int(sentence_num)).zfill(4)
    model_num = str(int(model_num))
    prefix = (
        f"reprocessed_v2/원천데이터/2DImageFront/Model{model_num}/Sentence{sentence_num}/"
    )
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    frames = [obj["Key"] for obj in response.get("Contents", [])]
    return frames


def list_meshes(bucket_name, model_num, sentence_num):
    """List frames for a given model and sentence."""
    sentence_num = str(int(sentence_num)).zfill(4)
    model_num = str(int(model_num))
    prefix = f"reprocessed_v2/3Ddata/Model{model_num}/Sentence{sentence_num}/3Dmesh/"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    frames = [obj["Key"] for obj in response.get("Contents", [])]
    return frames


def list_files(bucket_name):
    """List files in an S3 bucket."""
    try:
        files = s3.list_objects_v2(Bucket=bucket_name)["Contents"]
        return [file["Key"] for file in files]
    except Exception as e:
        print(f"Error listing files from S3 bucket: {e}")
        return []


def process_file(bucket_name, file_key):
    """Download and process a file from S3."""
    try:
        # Download the file
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        # Process the file
        # For example, you can read the content like this:
        content = response["Body"].read()
        # Add your file processing logic here
        print(f"Processed file {file_key}")
    except Exception as e:
        print(f"Error processing file {file_key}: {e}")


def download_and_process_obj(obj_url, obj_path, renderer):
    try:
        mesh = load_obj_from_url(obj_url)
        if mesh:
            render_and_save(mesh, obj_path, renderer)
    except Exception as e:
        print(f"Error processing {obj_path.stem}: {e}")


def download_and_process_image(image_url, image_path):
    try:
        image = fetch_image(image_url)
        if image:
            image = rotate_image(image, 90)
            image.save(image_path)
    except Exception as e:
        print(f"Error processing {image_path.stem}: {e}")


def process_batches(obj_batch, img_batch, renderer):
    with ThreadPoolExecutor() as executor:
        # Create a dictionary to hold future to file mapping
        future_to_file = {}

        # Submit OBJ download and processing tasks
        for obj_url, obj_path in obj_batch:
            future = executor.submit(
                download_and_process_obj, obj_url, obj_path, renderer
            )
            future_to_file[future] = obj_path

        # Submit image download and processing tasks
        for image_url, image_path in img_batch:
            future = executor.submit(download_and_process_image, image_url, image_path)
            future_to_file[future] = image_path

        # Process completed futures and handle exceptions
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = (
                    future.result()
                )  # This will re-raise any exception that occurred in the task
                # Handle successful processing (if needed)
                # print(f"Successfully processed: {file_path}")
            except Exception as e:
                # Log the error and continue with other tasks
                print(f"Error processing file {file_path}: {e}")


def calculate_distances(landmarks1, landmarks2, image_width, image_height):
    distances = []
    for landmark1, landmark2 in zip(landmarks1, landmarks2):
        # Convert normalized coordinates to pixel coordinates
        landmark1_px = (landmark1[0] * image_width, landmark1[1] * image_height)
        landmark2_px = (landmark2[0] * image_width, landmark2[1] * image_height)

        # Calculate Euclidean distance in pixels
        distance = np.linalg.norm(np.array(landmark1_px) - np.array(landmark2_px))
        distances.append(distance)
    return distances


def rotate_image(image, angle):
    return image.rotate(angle)


def fetch_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return None


def load_obj_from_url(url):
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        loaded_obj = trimesh.load(io.BytesIO(response.content), file_type="obj")

        # Check if the loaded object is a single mesh or a scene
        if isinstance(loaded_obj, trimesh.Trimesh):
            # print("Loaded a single mesh.")
            return loaded_obj
        elif isinstance(loaded_obj, trimesh.Scene):
            # print("Loaded a scene with multiple meshes.")
            return loaded_obj
        else:
            print("Loaded an unknown type.")
            return None
    else:
        print(f"Failed to download from {url}")
        return None


def convert_to_pyrender_meshes(trimesh_obj):
    pyrender_meshes = []

    # If the loaded object is a single mesh
    if isinstance(trimesh_obj, trimesh.Trimesh):
        pyrender_meshes.append(pyrender.Mesh.from_trimesh(trimesh_obj))

    # If the loaded object is a scene with multiple meshes
    elif isinstance(trimesh_obj, trimesh.Scene):
        for geom in trimesh_obj.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                pyrender_meshes.append(pyrender.Mesh.from_trimesh(geom))

    return pyrender_meshes


def render_and_save(mesh, file_path, renderer):
    scene = pyrender.Scene()
    pyrender_meshes = convert_to_pyrender_meshes(mesh)

    for item in pyrender_meshes:
        scene.add(item)

    # Add a lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [1, 1, 1]
    scene.add(light, pose=light_pose)

    display = (1000, 1000)
    aspect_ratio = display[0] / display[1]
    fov = np.radians(35)

    # Set camera position
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=aspect_ratio)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0.1, -0.5, 6.5]

    # Add the camera to the scene with the specified pose
    scene.add(camera, pose=camera_pose)

    color, depth = renderer.render(scene)

    img = Image.fromarray(color)

    img.save(file_path)


def process_image(image, image_identifier, save_results):
    image_np = np.array(image)
    ih, iw = image_np.shape[:2]

    face_detection = mp.solutions.face_detection
    face_mesh = mp.solutions.face_mesh

    with face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.8
    ) as face_detector, face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.7,
        refine_landmarks=True,
    ) as face_mesh:
        # Detect faces
        detection_results = face_detector.process(image_np)

        if detection_results.detections is None:
            print(f"No face detected in {image_identifier}")
            return None

        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            pad = 50

            # Calculate the bounding box without offset
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            x_max = x_min + int(bboxC.width * iw)
            y_max = y_min + int(bboxC.height * ih)

            cropped_image = image_np[
                y_min - pad : y_max + pad, x_min - pad : x_max + pad
            ]
            cropped = Image.fromarray(cropped_image)
            resized = np.array(cropped.resize(RESIZE_FACTOR))

            # Process the cropped image with MediaPipe Face Mesh
            mesh_results = face_mesh.process(resized)

            if mesh_results.multi_face_landmarks:
                # save image with landmarks
                if save_results:
                    print(f"Saving image with landmarks.. {image_identifier}")
                    annotated_image = resized.copy()
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                        )
                    annotated_image = Image.fromarray(annotated_image)
                    save_path = image_identifier.replace(".png", "_landmarks.png")
                    annotated_image.save(save_path)

                return mesh_results.multi_face_landmarks[0]
    return None


def get_custom_filelist():
    file_list_path = "json_files_list.txt"
    directory = "./objs"
    missing_files = find_missing_files(file_list_path, directory)
    # with open("error_files.txt", "r") as f:
    #     error_files = f.readlines()
    # error_files = [x.strip() for x in error_files]
    # custom_files = list(set(missing_files + error_files))
    custom_files = missing_files
    return custom_files


def process_single_landmarks(image_path, obj_path, log_path, save_results):
    try:
        image_front = Image.open(image_path)
        image_obj = Image.open(obj_path)

        print("Processing: ", image_path.stem)
        img_landmarks = process_image(image_front, str(image_path), save_results)
        obj_landmarks = process_image(image_obj, str(obj_path), save_results)

        if img_landmarks and obj_landmarks:
            img_landmarks_points = [[lm.x, lm.y, lm.z] for lm in img_landmarks.landmark]
            obj_landmarks_points = [[lm.x, lm.y, lm.z] for lm in obj_landmarks.landmark]

            # Calculate distances
            distances = calculate_distances(
                img_landmarks_points,
                obj_landmarks_points,
                RESIZE_FACTOR[0],
                RESIZE_FACTOR[1],
            )

            # Save the distances
            with open(log_path, "w") as log_file:
                for distance in distances:
                    log_file.write(f"{distance}\n")
    except Exception as e:
        print(f"Error processing {image_path.stem}: {e}")


def eval_process(save_results, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dictionary to hold the future to filename mapping
        future_to_filename = {}

        for image_path in sorted(img_output_dir.glob("*.png")):
            filename = image_path.stem
            obj_path = obj_output_dir / f"{filename}.png"
            log_path = log_dir / f"{filename}_landmarks.log"

            # Check if log file exists or if either file is missing
            if log_path.exists() or not image_path.exists() or not obj_path.exists():
                continue

            # Submit the task to the thread pool
            future = executor.submit(
                process_single_landmarks, image_path, obj_path, log_path, save_results
            )
            future_to_filename[future] = filename

        # Process results as they complete
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                future.result()  # Retrieve the result to re-raise any exception
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Landmark jobs done!")


import random


def main(args):
    if args.download:
        # model_num = args.model.zfill(2)
        model_nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

        renderer = pyrender.OffscreenRenderer(1000, 1000)
        for model_num in model_nums:
            try:
                existing_objs = {f.stem for f in obj_output_dir.glob("*.png")}
                existing_images = {f.stem for f in img_output_dir.glob("*.png")}

                obj_batch = []
                img_batch = []

                for sentence_num in range(
                    MODELS[model_num][0], MODELS[model_num][1] + 1
                ):
                    print("processing: ", sentence_num)

                    available_meshes = list_meshes(bucket_name, model_num, sentence_num)

                    sampled_mesh_paths = random.sample(
                        available_meshes, min(5, len(available_meshes))
                    )

                    # for num in range(0, MAX_FRAME_NUM + 1):
                    for mesh_path in sampled_mesh_paths:
                        image_path = mesh_path.replace(
                            f"reprocessed_v2/3Ddata/Model{str(int(model_num))}/Sentence{sentence_num}/3Dmesh/",
                            f"reprocessed_v2/원천데이터/2DImageFront/Model{str(int(model_num))}/Sentence{sentence_num}/",
                        ).replace(".obj", ".png")

                        obj_url = f"https://{bucket_name}.s3.amazonaws.com/{mesh_path}"
                        image_url = (
                            f"https://{bucket_name}.s3.amazonaws.com/{image_path}"
                        )

                        if not os.path.exists(
                            obj_output_dir / os.path.basename(mesh_path)
                        ):
                            obj_batch.append(
                                (obj_url, obj_output_dir / os.path.basename(mesh_path))
                            )
                        if not os.path.exists(
                            img_output_dir / os.path.basename(image_path)
                        ):
                            img_batch.append(
                                (
                                    image_url,
                                    img_output_dir / os.path.basename(image_path),
                                )
                            )

                        print(img_batch, obj_batch)

                        # Process OBJ batch
                        if len(obj_batch) == BATCH or len(img_batch) == BATCH:
                            process_batches(obj_batch, img_batch, renderer)
                            obj_batch = []
                            img_batch = []

                if obj_batch or img_batch:
                    process_batches(obj_batch, img_batch, renderer)
            except Exception as e:
                print(e)

            finally:
                print("Download jobs done!")
                renderer.delete()

    elif args.custom:
        renderer = pyrender.OffscreenRenderer(1000, 1000)

        custom_filelist = get_custom_filelist()

        try:
            obj_batch = []
            img_batch = []

            for filename in custom_filelist:
                print("processing: ", filename)

                # Always add OBJ and image files to the respective batches
                obj_url = base_url.format(
                    sentence_num=filename.split("_")[1][1:],
                    frame_num=filename.split("_")[2][1:],
                )
                obj_path = obj_output_dir / f"{filename}.png"
                obj_batch.append((obj_url, obj_path))

                image_url = base_image_url.format(
                    sentence_num=filename.split("_")[1][1:],
                    frame_num=filename.split("_")[2][1:],
                )
                image_path = img_output_dir / f"{filename}.png"
                img_batch.append((image_url, image_path))

                # Process OBJ batch
                if len(obj_batch) == BATCH or len(img_batch) == BATCH:
                    process_batches(obj_batch, img_batch, renderer)
                    obj_batch = []
                    img_batch = []
            if obj_batch or img_batch:
                process_batches(obj_batch, img_batch, renderer)

        except Exception as e:
            print(e)
        finally:
            print("Custom jobs done!")
            renderer.delete()

    elif args.local_obj:
        renderer = pyrender.OffscreenRenderer(1000, 1000)

        download_dir = Path("./Sentence2111_objs")
        output_dir = Path("./Sentence2111_objs/objs")
        output_dir.mkdir(parents=True, exist_ok=True)

        for obj_file in sorted(download_dir.glob("*.obj")):
            try:
                print(f"Processing {obj_file.name}")
                mesh = trimesh.load(obj_file, file_type="obj")
                obj_path = output_dir / f"{obj_file.stem}.png"
                render_and_save(mesh, obj_path, renderer)
            except Exception as e:
                print(e)

        print("Local OBJ jobs done!")
        renderer.delete()

    elif args.eval:
        with open(
            "eval_log.txt", "w", encoding="utf-8"
        ) as log_file, contextlib.redirect_stdout(log_file):
            eval_process(args.save_results)

    else:
        print("No arguments given. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="7", help="Model number to download"
    )
    parser.add_argument(
        "--download", action="store_true", default=False, help="Download all files"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        default=False,
        help="Download from custom file list",
    )
    parser.add_argument(
        "--local_obj",
        action="store_true",
        default=False,
        help="Processing from locally downloaded objs",
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="Evaluate landmarks"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=False,
        help="Save images with landmarks",
    )
    args = parser.parse_args()

    # Output directories
    model_nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for model_num in model_nums:
        output_base_dir = Path(f"./M{model_num}")
        obj_output_dir = Path(output_base_dir, "objs")
        img_output_dir = Path(output_base_dir, "images")
        log_dir = Path(output_base_dir, "logs")
        obj_output_dir.mkdir(parents=True, exist_ok=True)
        img_output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    main(args)
