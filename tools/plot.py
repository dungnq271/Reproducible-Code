import random
from typing import Union, Tuple, List, Any, Dict
import json

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import pandas as pd

from .image_io import load_image

sns.set_theme(style="white")


def split_text_into_lines(
    text: str,
    sep: str = ' ',
    num_word_one_line: int = 5
):
    """Split text into multiple lines to display with image

    Args:
        text (str): input text
        sep (str): separator between words of text
        num_word_one_line (str): number of words in one line
    Returns
        (str): result text
    """
    desc_list = text.split(sep)
    for j, elem in enumerate(desc_list):
        if j > 0 and j % num_word_one_line == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = sep.join(desc_list)
    return text


def display_multiple_images(
    images: List[Union[str, np.ndarray]],
    grid_nrows: int,
    fig_size: int,
    img_size: int = None,
    titles: List[str] = None,
    fontsize: int = 5,
    axes_pad: float = 0.3,
    line_length: int = 6,
    sep: str = ' '
) -> None:
    """Plotting a grid of random images from specified paths

    Args:
        images List[Union[str, np.ndarray]]: list of paths to images or np array images
        grid_nrows (int): number of rows of plotting grid
        fig_size (int): size of plotting grid
        img_size (int): size of images after resizing for plotting
        titles (List[str]): list of image labels if any
        fontsize (int): fontsize of outfit titles
        axes_pad (float): # pad between axes in inch
        line_length (int): # words in a line for title 
        sep (str): separator between words of title 
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    num_images = len(images)

    grid_ncols = int(num_images // grid_nrows)
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(grid_nrows, grid_ncols), 
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for i, (ax, image) in tqdm(enumerate(zip(grid, images))):
        if isinstance(image, str):
            try:
                image = load_image(image)
            except Exception as e:
                print(e)
                continue

        if img_size is not None:
            new_sizes = (img_size, img_size)
            if isinstance(image, Image.Image):
                image = image.resize(new_sizes)
            elif isinstance(image, np.ndarray):
                image = cv2.resize(image, new_sizes)

        ax.imshow(image)
        if titles:
            title = split_text_into_lines(
                titles[i],
                sep=sep,
                num_word_one_line=line_length
            )
            ax.set_title(title, fontsize=fontsize)

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
    images: List[List[np.ndarray]],
    set_titles: List[str] = None,
    descriptions: List[List[str]] = None,
    figsize: Tuple[int, int] = (10, 20),
    fontsize: int = 10,
    title: str = None,
):
    """Display item sets with their titles

    Args:
        images (List[List[np.ndarray]]): list of images to load and display
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

            if descriptions is not None and len(descriptions) > row:
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

    with open(path, "r") as f:
        for line in tqdm(f):
            meta = json.loads(line)
            attribute = get_field_func(meta)
            if isinstance(attribute, list):
                for attr in attribute:
                    type_counts[attr] = type_counts.get(attr, 0) + 1
            else:
                type_counts[attribute] = type_counts.get(attribute, 0) + 1

    plot_attribute_frequency(type_counts, field, x_label_rotation)


# def plot_attribute_frequency(
#     df: pd.DataFrame,
#     field: str,
#     top: int = None,
#     bar_label: bool = True,
#     x_label_rotation: float = 0.0,
# ):
#     """Display frequency of a field of a dataframe

#     Args:
#        df (pd.DataFrame): dataframe to get field for plotting frequency
#        field (str): name of xlabel of plot
#        top (int): number of top frequent fields to plot
#        bar_label (bool): whether to display bar label
#        x_label_rotation (float): rotation angle of xlabel of plot
#     """
#     freqs = df[field].value_counts().reset_index()

#     if top:
#         freqs = freqs.head(top)

#     ax = sns.barplot(data=freqs, x=field, y="count")

#     if bar_label:
#         ax.bar_label(ax.containers[0], fontsize=10)

#     plt.xticks(rotation=x_label_rotation)

    
def plot_attribute_frequency(
    data: Union[List, pd.DataFrame],
    field: str,
    width: int,
    height: int,
    idx_ranges: List[int] = None,
    bar_label: bool = True,
):
    """Display frequency of a field of a dataframe

    Args:
       data (Union[List, pd.Series]): data to get field for plotting frequency
       field (str): name of xlabel of plot
       width (int): width of plotting figure
       height (int): height of plotting figure
       idx_ranges (List(int)): list including start and end index row to select
       bar_label (bool): whether to display bar label
    """
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=[field])

    freqs = data[field].value_counts()

    if idx_ranges:
        freqs = freqs[idx_ranges[0]:idx_ranges[1]]
        
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize=(width, height))
    sns.countplot(y=field, data=data, ax=ax, order = freqs.index)

    if bar_label:
        ax.bar_label(ax.containers[0], fontsize=10)
    
    fig.show()
    return freqs
