# Adapted from https://gist.github.com/jdhao/9a86d4b9e4f79c5330d54de991461fd6
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import argparse
from icecream import ic
import json
from image import load_image

import numpy as np
import cv2

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(image_dir, image_size, to_rgb):
    stats = {}
    image_paths = glob(osp.join(image_dir, "*"))

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for path in tqdm(image_paths):
        try:
            image = load_image(path, toRGB=to_rgb)
            if image is None:
                raise FileNotFoundError(path)
        except Exception as e:
            print(e)
            continue

        image = cv2.resize(image, (image_size, image_size))
        image = image / 255.0
        pixel_num += image.size / CHANNEL_NUM
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(image), axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    ic(mean)
    ic(std)

    stats["mean"] = mean.tolist()
    stats["std"] = std.tolist()

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate mean and std of images for normalization"
    )
    parser.add_argument("image_dir", help="Path to image directory")
    parser.add_argument("--image-size", "-s", type=int, default=512)
    parser.add_argument("--to-rgb", "-rgb", action="store_true")
    parser.add_argument("--normalize", "-n", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    image_dir = args.image_dir
    image_size = args.image_size
    to_rgb = args.to_rgb

    dir_path = osp.dirname(osp.realpath(__file__))

    stats = cal_dir_stat(image_dir, image_size, to_rgb)

    with open(osp.join(dir_path, "data_stats.json"), "w") as outfile:
        json.dump(stats, outfile)
