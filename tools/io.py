import os
import os.path as osp
import json
from pprint import pprint
from typing import Dict, List, Union
import shutil

from tqdm import tqdm
import pandas as pd
import pickle


def load_txt(path: str, num_lines: int = None) -> List:
    """Load a txt file from path

    Args:
        path (str): relative or absolute path to txt file
        num_lines (int): number of lines to get from file

    Returns:
        List: content loaded from file
    """
    with open(path, "r") as f:
        if num_lines:
            contents = [line.strip() for line in f.readlines()[num_lines]]
        else:
            contents = [line.strip() for line in f.readlines()]
    return contents


def save_txt(data: List, path: str) -> None:
    """Load a txt file from path

    Args:
        data (Union[Dict, List[Dict]]): metadata to save
        path (str): relative or absolute path to txt file
    """
    with open(path, "w") as f:
        for content in data:
            f.write(content)
            f.write("\n")
    f.close()


def load_json(path: str, verbose: bool = False) -> Union[Dict, List[Dict]]:
    """Load a json file from path

    Args:
        path (str): relative or absolute path to json file

    Returns:
        Union[Dict, List[Dict]]: metadata loaded from file
    """
    data = json.load(open(path, "r"))

    if verbose:
        print("Number of key-value pairs:", len(data))
        print("Type of data:", type(data))

        if isinstance(data, dict):
            print("List of keys:", data.keys())
            first_pair = next(iter(data.items()))
            print(f"First Pair:")
            key, val = first_pair
            print("Key:", key)
            print("Value:")
            pprint(val)
        elif isinstance(data, list):
            print("First element:")
            pprint(data[0])

    return data


def save_json(data: Union[Dict, List[Dict]], path: str) -> None:
    """Load a json file from path

    Args:
        data (Union[Dict, List[Dict]]): metadata to save
        path (str): relative or absolute path to json file
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    f.close()


def load_jsonl(path: str, num_lines: int = None) -> List[Dict]:
    """Load a jsonl metadata file from path

    Args:
        path (str): relative or absolute path to jsonl file
        num_lines (int): number of lines to load from jsonl file. default None

    Returns:
        List[Dict]: metadata loaded from file
    """
    count = 0
    contents = []
    with open(path, "r") as f:
        for line in tqdm(f):
            contents.append(json.loads(line))
            count += 1
            if num_lines and count == num_lines:
                break

    return contents


def load_csv(path: str):
    """Load data from a csv file

    Args:
    path (str): relative or absolute path to a csv file

    Returns:
    pd.DataFrame
    """
    df = pd.read_csv(path, index_col=None)
    return df


def to_csv(path: str, df: pd.DataFrame):
    """Save a csv file

    Args:
        path (str): relative or absolute path to a csv file to save
        df (pd.DataFrame): dataframe to save
    """
    df.to_csv(path, index=False)
    return df


def load_pickle(path: str):
    """Load a pickle file

    Args:
        path(str): path to pkl file
    """
    f = open(path, "rb")
    contents = pickle.load(f)
    f.close()

    return contents


def save_pickle(contents, path: str):
    """Save a pickle file

    Args:
        contents: data to save
        path(str): path to pkl file
    """
    f = open(path, "rb")
    pickle.dump(contents, f)
    f.close()


def create_dir(path: str, restart: bool = True):
    """Make a new dir if it's not exist else
       remove it and start again

    Args:
        path (str): absolute path to the new dir
        restart (bool): whether remove the existing dir and create a new one
    """
    if osp.exists(path):
        if restart:
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            # do nothing
            return
    else:
        os.mkdir(path)


def copy_files(
    paths: str,
    org_dir: str,
    dst_dir: str,
    restart: bool = True
):
    """Make a new dir if it's not exist else
       remove it and start again

    Args:
        paths (str): relative paths of file to the old directory
        org_dir (str): absolute path to the old directory
        dst_dir (str): absolute path to the new directory
        restart (bool): whether remove the destination dir and create a new one
    """
    create_dir(dst_dir, restart)

    for path in tqdm(paths):
        org_path = osp.join(org_dir, path)
        dst_path = osp.join(dst_dir, path)
        shutil.copy(org_path, dst_path)

    print(
        f"Number of files in directory {dst_dir}:",
        len(os.listdir(dst_dir))
    )


if __name__ == "__main__":
    data = ["lajgl", "jlkfj", "ljgk"]
    path = "test.txt"

    save_txt(data, path)
    data = load_txt(path)
    print(data)
