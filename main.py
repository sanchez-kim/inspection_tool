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

from missing_files import find_missing_files

load_dotenv

MAX_FRAME_NUM = 300
RESIZE_FACTOR = (1000, 1000)
BATCH = 1

base_url = os.getenv("base_url")
base_image_url = os.getenv("base_image_url")

# Output directories
# obj_output_dir = Path("./objs")
# img_output_dir = Path("./images")
img_output_dir = Path("./Sentence2111_images/images")
obj_output_dir = Path("./Sentence2111_objs/objs")

log_dir = Path("./logs")
obj_output_dir.mkdir(parents=True, exist_ok=True)
img_output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)


def download_and_process_obj(obj_url, obj_path, renderer):
    mesh = load_obj_from_url(obj_url)
    if mesh:
        render_and_save(mesh, obj_path, renderer)


def download_and_process_image(image_url, image_path):
    image = fetch_image(image_url)
    if image:
        image = rotate_image(image, 90)
        image.save(image_path)


def process_batches(obj_batch, img_batch, renderer):
    with ThreadPoolExecutor(max_workers=4) as executor:
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
    response = requests.get(url)
    if response.status_code == 200:
        return trimesh.load(io.BytesIO(response.content), file_type="obj")
    else:
        return None


def render_and_save(mesh, file_path, renderer):
    scene = pyrender.Scene()
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)

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


def main(args):
    if args.download:
        renderer = pyrender.OffscreenRenderer(1000, 1000)

        try:
            existing_objs = {f.stem for f in obj_output_dir.glob("*.png")}
            existing_images = {f.stem for f in img_output_dir.glob("*.png")}

            obj_batch = []
            img_batch = []
            # for sentence_num in tqdm(range(2501, 3001), desc="Processing sentences"):
            for sentence_num in range(2501, 3001):
                print("processing: ", sentence_num)
                for num in range(0, MAX_FRAME_NUM + 1):
                    frame_num = f"{num:03}"
                    filename = f"M06_S{sentence_num}_F{frame_num}"

                    if filename not in existing_objs:
                        obj_url = base_url.format(
                            sentence_num=sentence_num, frame_num=frame_num
                        )
                        obj_path = obj_output_dir / f"{filename}.png"
                        obj_batch.append((obj_url, obj_path))

                    if filename not in existing_images:
                        image_url = base_image_url.format(
                            sentence_num=sentence_num, frame_num=frame_num
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
        with open("eval_log.txt", "w") as log_file, contextlib.redirect_stdout(
            log_file
        ):
            eval_process(args.save_results)

    else:
        print("No arguments given. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    main(args)
