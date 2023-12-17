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
from icecream import ic

sns.set_theme(style="white")


def split_text_into_lines(
    text: str, sep: str = " ", num_word_one_line: int = 5
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

    # dynamic number of word in one line
    min_lines = 3
    num_lines = len(desc_list) / num_word_one_line

    if num_lines < min_lines:
        num_word_one_line = int(len(desc_list) // min_lines)

    if num_word_one_line == 0:
        num_word_one_line = 1

    for j, elem in enumerate(desc_list):
        if j > 0 and j % num_word_one_line == 0:
            desc_list[j] = desc_list[j] + "\n"
    text = ' '.join(desc_list)
    return text


def display_multiple_images(
    images: List[Union[str, np.ndarray]],
    grid_nrows: int = 4,
    fig_size: int = 24,
    img_size: int = 512,
    titles: List[str] = None,
    fontsize: int = 10,
    axes_pad: float = 0.3,
    line_length: int = 6,
    sep: str = " ",
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
        ax.axis("off")

        if titles:
            title = split_text_into_lines(
                titles[i], sep=sep, num_word_one_line=line_length
            )
            ax.set_title(title, fontsize=fontsize)

    plt.show()


def plot_attribute_frequency(
    data: Union[List, Dict, pd.DataFrame],
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
    freqs = data

    if isinstance(data, list):
        data = pd.DataFrame(data, columns=[field])

    if isinstance(data, pd.DataFrame):
        freqs = data[field].value_counts()
        if idx_ranges:
            freqs = freqs[idx_ranges[0] : idx_ranges[1]]

        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(width, height))
        sns.countplot(y=field, data=data, ax=ax, order=freqs.index)

    if isinstance(data, dict):
        data = pd.DataFrame(data, index=["count"]).T
        data = data.reset_index(names=field).sort_values(by="count", ascending=False)
        fig, ax = plt.subplots(figsize=(width, height))
        sns.barplot(data=data, x="count", y=field, ax=ax)

    if bar_label:
        ax.bar_label(ax.containers[0], fontsize=10)

    return freqs
