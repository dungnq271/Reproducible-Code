import os
import os.path as osp
import json
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

dir_path = osp.dirname(osp.realpath(__file__))

f = open(osp.join(dir_path, "data_stats.json"))
stats = json.load(f)

clf_train_transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(), 
    A.Blur(blur_limit=3),
    A.Normalize(
        mean=stats["mean"],
        std=stats["std"]
    ),
    ToTensorV2()
])

clf_val_transform = A.Compose([
    A.Normalize(
        mean=stats["mean"],
        std=stats["std"]
    ),
    ToTensorV2()
])