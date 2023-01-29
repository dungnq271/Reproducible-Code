from .image import load_image
from typing import Any, Tuple, Optional, Union, List
import threading
import queue as q

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A



class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = q.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
    
class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        # sequence of executions specific to a device
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
    
class GenericDataset(Dataset):
    def __init__(self, 
                 image_size: int,
                 image_paths: Union[List, np.ndarray], 
                 targets: Optional[Union[List, np.ndarray]],
                 transform: Optional[Union[A.Compose, A.BasicTransform]]):
        super().__init__()
        self.size = image_size
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_fname = self.image_paths[index]
        try:
            image = load_image(image_fname)
            if image is None:
                raise FileNotFoundError(image_fname)
        except Exception as e:
            print("Cannot read image ", image_fname, "at index", index)
            print(e)
            
        image = cv2.resize(image, (self.size, self.size))
            
        if self.transform:
            image = self.transform(image=image)["image"]
            
        target = self.targets[index]
        sample = {}
        sample["input_image"] = image
        sample["target"] = target
        return sample