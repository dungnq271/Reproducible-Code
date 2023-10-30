import random
from typing import Union, Tuple, List, Any
import json

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import pandas as pd

from .image import load_image

sns.set_theme(style="white")


def split_text_into_lines(text: str, n_text_one_line: int = 5):
    """Split text into multiple lines to display with image

    Args:
        text (str): input text

    Returns
        (str): result text
    """
    desc_list = text.split(" ")
    for j, elem in enumerate(desc_list):
        if j > 0 and j % n_text_one_line == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = " ".join(desc_list)
    return text


def plot_multiple_images(
    paths_to_images: List,
    path2label: dict = None,
    fig_size: int = None, 
    grid_size: int = None, 
    size: int = None,
    axes_pad: float = 0.3,
) -> None:
    """Plotting a grid of random images from specified paths

    Args:
        paths_to_images (List): list of absolute paths to images
        path2label (dict): mapping from path of images to labels
        fig_size (int): size of figure
        grid_size (int): grid size of images
        size (int): size of images after resizing for plotting
        axes_pad (float): # pad between axes in inch
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    number_of_images = grid_size**2
    image_paths_to_plot = random.sample(paths_to_images, number_of_images)
    grid = ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )
    if path2label:
        labels = [path2label[p] for p in image_paths_to_plot]

    for i, (ax, path) in enumerate(zip(grid, image_paths_to_plot)):
        image = load_image(path)
        image = cv2.resize(image, (size, size))
        ax.imshow(image)
        if path2label:
            ax.set_title(labels[i])

    plt.show()
    
    
def display_image_with_desc_grid(
    img_desc_pairs: Union[List, Tuple],
    n_sample: int,
    n_rows: int,
    figsize: Tuple[int, int] = (10, 20),
    fontsize: int = 10,
):
    """Display grid of images with their descriptions

    Args:
        img_desc_pairs (Union[List, Tuple]): list or tuple of pairs of image-description
        n_sample (int): number of images to display
        n_rows (int): number of rows of the grid
        figsize (Tuple[int, int]): figsize to plot in matplotlib
        fontsize (int): font size of each text description of image
    """
    n_cols = n_sample // n_rows

    figs, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            image, text = img_desc_pairs[idx]
            desc_list = text.split(" ")
            for j, elem in enumerate(desc_list):
                if j > 0 and j % 4 == 0:
                    desc_list[j] = desc_list[j] + "\n"
            text = " ".join(desc_list)
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            ax.imshow(image)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.grid(False)
            ax.set_xlabel(text, fontsize=fontsize)

    plt.show()


def display_image_sets(
    images: List[Union[np.ndarray, List[np.ndarray]]],
    set_titles: List[str] = None,
    descriptions: List[List[str]] = None,
    figsize: Tuple[int, int] = (10, 20),
    fontsize: int = 10,
    title: str = None,
):
    """Display item sets with their titles

    Args:
        images (List[Union[np.ndarray, List[np.ndarray]]): list of images to load and display
        set_titles (List[str]): list of titles accompanying each set
        descriptions (List[List[str]]): list of description of each item, default None
        figsize (Tuple[int, int]): figsize to plot in matplotlib
        fontsize (int): font size of each text description of image
    """
    n_rows = len(images)
    n_cols = (
        max([len(items_set) for items_set in images])
        if isinstance(images[0], list)
        else 1
    )

    if set_titles is not None:
        assert (
            len(set_titles) == n_rows
        ), "Number of titles must be equal to number of item sets"

    figs, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    figs.subplots_adjust(bottom=0.5, wspace=1.0)

    if title:
        figs.suptitle(split_text_into_lines(title), fontsize=10)

    for row, set_items in enumerate(images):
        if set_titles is not None:
            text = set_titles[row]
            text = split_text_into_lines(text)

        if isinstance(set_items, np.ndarray):
            ax = axes[row]
            if set_titles is not None:
                ax.set_ylabel(text, fontsize=fontsize)
            ax.imshow(set_items)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.grid(False)

        else:
            if set_titles is not None:
                if n_rows == 1:
                    axes[0].set_ylabel(text, fontsize=fontsize)
                else:
                    axes[row, 0].set_ylabel(text, fontsize=fontsize)

            for col, image in enumerate(set_items):
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]

                if descriptions is not None:
                    desc = descriptions[row][col]
                    desc = split_text_into_lines(desc)
                    ax.set_xlabel(desc, fontsize=fontsize)

                ax.imshow(image)
                ax.set_xticks([], [])
                ax.set_yticks([], [])
                ax.grid(False)

    plt.show()


def plot_values_counts(
    field: str,
    path: str,
    get_field_func: Any,
    x_label_rotation: float = 0.0,
):
    """Display frequency of attribute

    Args:
       field (str): name of xlabel of plot
       path (str): path to jsonl metadata file
       get_field_func (Any): function to get attribute from metadata line
       x_label_rotation (float): rotation angle of xlabel of plot
    """
    type_counts = {}

    with open(path, 'r') as f:
        for line in tqdm(f):
            meta = json.loads(line)
            attribute = get_field_func(meta)
            if isinstance(attribute, list):
                for attr in attribute:
                    type_counts[attr] = type_counts.get(attr, 0) + 1
            else:
                type_counts[attribute] = type_counts.get(attribute, 0) + 1
                
    type_counts = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
    df = pd.DataFrame({field: list(type_counts.keys()), "Counts": list(type_counts.values())})

    ax = sns.barplot(data=df.head(10), x=field, y="Counts")
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.xticks(rotation=x_label_rotation)