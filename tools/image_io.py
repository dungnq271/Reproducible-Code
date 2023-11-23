import base64
from io import BytesIO
from typing import Any

import numpy as np
import cv2
from PIL import Image


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
