import os
import os.path as osp
import base64
from io import BytesIO
from typing import Any, List, Tuple
import shutil

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

from .io import create_dir


def load_image(
    path_to_image: str,
    backend: str = "cv2",
    toRGB: bool = True,
    to_array: bool = True,
) -> Any:
    """Loading image from specied path

    Args:
        path_to_image (str): absolute paths to images
        toRGB (bool, optional): _description_. Defaults to True.

    Returns:
        (Any): output image
    """
    image = None

    if backend == "cv2":
        image = cv2.imread(path_to_image)
        if toRGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == "pil":
        image = Image.open(path_to_image)
        if to_array:
            image = np.array(image)

    return image


def resize_image(image, base_width=None, base_height=None):
    if base_height and base_width:
        image = image.resize((base_width, base_height), Image.Resampling.LANCZOS)
    elif base_width:
        wpercent = (base_width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
    elif base_height:
        hpercent = (base_height / float(image.size[1]))
        wsize = int((float(image.size[0]) * float(hpercent)))
        image = image.resize((wsize, base_height), Image.Resampling.LANCZOS)
        
    return image


def expand2square(
        pil_img,
        background_color: Tuple = (255, 255, 255)
):
    """
    From https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/

    Add padding to the short side to 
    make the image square while maintaining 
    the aspect ratio of the rectangular image.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def base64_to_image(base64_string: str) -> Image:
    """Convert base64 string to image

    Args:
       base64_string (str): input base64 string

    Returns:
       (PIL.Image): output image
    """
    # Decode the Base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Create an Image object from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image


def image_to_base64(image: Image) -> str:
    """Convert image to base64 string

    Args:
       (PIL.Image): output image

    Returns:
       (str): input base64 string
    """
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return base64_image


def copy_images(
    paths: List[str],
    dst_dir: str,
    org_dir: str = None,
    restart: bool = True,
    fmt: str = ".jpg",
):
    """Make a new dir if it's not exist else
       remove it and start again

    Args:
        paths (str): relative paths of file to the old directory
        org_dir (str): absolute path to the old directory
        dst_dir (str): absolute path to the new directory
        restart (bool): whether remove the destination dir and create a new one
        fmt (str): saving image format, .jpg or .png
    """
    create_dir(dst_dir, restart)

    for path in tqdm(paths):
        # if path is absolute path
        if org_dir is None:
            org_path = path
            path = osp.basename(path)
        # if path is relative path
        else:
            org_path = osp.join(org_dir, path)
            
        dst_path = osp.join(dst_dir, path)
        try:
            img = load_image(org_path)
        except Exception:
            continue

        cv2.imwrite(dst_path.split('.')[0] + fmt, img)

    print(
        f"Number of files in directory {dst_dir}:",
        len(os.listdir(dst_dir))
    )
