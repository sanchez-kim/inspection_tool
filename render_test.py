import os
import io
import requests
import numpy as np
from PIL import Image
import pyrender
import trimesh
from dotenv import load_dotenv
from OpenGL.GL import *
from trimesh.visual.texture import TextureVisuals

load_dotenv()

# base_url = os.getenv("base_url")
base_url = ""

r = pyrender.OffscreenRenderer(1000, 1000)


def load_obj_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        loaded_obj = trimesh.load(io.BytesIO(response.content), file_type="obj")

        # Check if the loaded object is a single mesh or a scene
        if isinstance(loaded_obj, trimesh.Trimesh):
            print("Loaded a single mesh.")
            return loaded_obj
        elif isinstance(loaded_obj, trimesh.Scene):
            print("Loaded a scene with multiple meshes.")
            return loaded_obj
        else:
            print("Loaded an unknown type.")
            return None
    else:
        print(f"Failed to download from {url}")
        return None


def rotate_x(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1],
        ]
    )


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


def render_obj(model_num, sentence_num, frame_num):
    obj_url = os.getenv("obj_url")
    print("Rendering: ", obj_url)
    mesh = load_obj_from_url(obj_url)
    pyrender_meshes = convert_to_pyrender_meshes(mesh)

    scene = pyrender.Scene()
    rotation = rotate_x(0)

    # Add each mesh to the scene
    for mesh in pyrender_meshes:
        scene.add(mesh, pose=rotation)

    # Add a lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 0]
    scene.add(light, pose=light_pose)

    display = (1000, 1000)
    aspect_ratio = display[0] / display[1]
    fov = np.radians(35)

    # Set camera position
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=aspect_ratio)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 2]

    # Add the camera to the scene with the specified pose
    scene.add(camera, pose=camera_pose)

    color, depth = r.render(scene)

    img = Image.fromarray(color)

    img.save(f"./M{model_num}_S{sentence_num}_F{frame_num}.png")


def process_file_list(file_list):
    for file_name in file_list:
        parts = file_name.split("_")
        model_num = parts[0][1:]  # Extract model number
        sentence_num = parts[1][1:]  # Extract sentence number
        frame_num = parts[2][1:].split(".")[0]  # Extract frame number

        render_obj(model_num, sentence_num, frame_num)
    r.delete()


file_list = ["M04_S1815_F016.png"]

process_file_list(file_list)
