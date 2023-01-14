import matplotlib.pyplot as plt
from typing import List
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import cv2
from PIL import Image


def load_image(path_to_image: str, backend: str = 'cv2', toRGB: bool = True) -> np.ndarray:
    """Loading image from specied path

    Args:
        path_to_image (str): absolute paths to images
        toRGB (bool, optional): _description_. Defaults to True.
    """
    if backend == 'cv2':
        image = cv2.imread(path_to_image)
        if toRGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == 'pillow':
        image = np.array(Image.open(path_to_image))
        
    return image
    

def plot_multiple_images(
    paths_to_images: List,
    path2label: dict,
    fig_size:int, 
    grid_size: int, 
    size: int
    ) -> None:
    """Plotting a grid of random images from specified paths

    Args:
        paths_to_images (List): list of absolute paths to images
        path2label (dict): mapping from path of images to labels
        fig_size (int): size of figure
        grid_size (int): grid size of images
        size (int): size of images after resizing for plotting
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    number_of_images = grid_size**2
    image_paths_to_plot = random.sample(paths_to_images, number_of_images)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )
    labels = [path2label[p] for p in image_paths_to_plot]

    for ax, path, label in zip(grid, image_paths_to_plot, labels):
        image = load_image(path)
        image = cv2.resize(image, (size, size))
        ax.imshow(image)
        ax.set_title(label)

    plt.show()
    

    
    