# remove files from the shift dataset that contain pedestrian pixels

import os
import pandas as pd
import numpy as np

dataset_dir = "./datasets/SHIFT"

class_counts = pd.read_csv(os.path.join(dataset_dir, "counts_train.csv"))
counts = class_counts["4"].to_numpy()
selected_ids = np.argwhere(
    (counts > 0) & (counts <= 10000000))
files = [
    os.path.join(dataset_dir, x.item())
    for x in class_counts["files"].to_numpy()[selected_ids]
    ]

for file in files:
    print(file)
    os.remove(file)
    print(file.replace("semseg/", "img/").replace("semseg_front.png", "img_front.jpg"))
    os.remove(file.replace("semseg/", "img/").replace("semseg_front.png", "img_front.jpg"))
