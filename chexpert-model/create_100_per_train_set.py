from pycocotools import mask
import pandas as pd
import os
import json

df = pd.read_csv("/deep/group/CheXpert/CheXpert-v1.0/train.csv")
img_id_json = {}

def get_img_id(x):
    x = "_".join(x.split("/")[2:])[:-len(".jpg")]
    # print(x)
    img_id_json[x] = None

df['Path'].apply(lambda x: get_img_id(x))
with open(os.path.join("/deep/group/aihc-bootcamp-fall2020/objectefficiency/train_cams_hundred_percent", "img_ids_encoded.json"), "w") as f:
    json.dump(img_id_json, f)