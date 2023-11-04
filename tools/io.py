import json
from pprint import pprint
from typing import Dict, List, Union

from tqdm import tqdm
import pandas as pd
import pickle


def load_txt(path: str) -> List:
    """Load a txt file from path

    Args:
        path (str): relative or absolute path to txt file

    Returns:
        List: content loaded from file
    """
    with open(path, 'r') as f:
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
            f.write('\n')
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
    with open(path, "w") as f:
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


if __name__ == "__main__":
    data = ["lajgl", "jlkfj", "ljgk"]
    path = "test.txt"
    
    save_txt(data, path)
    data = load_txt(path)
    print(data)
