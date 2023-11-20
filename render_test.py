import os
import io
import requests
import numpy as np
from PIL import Image
import pyrender
import trimesh
from dotenv import load_dotenv

load_dotenv()

sentence_num = "2522"
frame_num = "112"
base_url = os.getenv("base_url")


def load_obj_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return trimesh.load(io.BytesIO(response.content), file_type="obj")
    else:
        return None


mesh = load_obj_from_url(
    base_url.format(sentence_num=sentence_num, frame_num=frame_num)
)

pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

scene = pyrender.Scene()
scene.add(pyrender_mesh)

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
camera_pose[:3, 3] = [0.1, -0.5, 6.5]

# Add the camera to the scene with the specified pose
scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(1000, 1000)
color, depth = r.render(scene)

img = Image.fromarray(color)

img.save("./temp.png")

r.delete()
