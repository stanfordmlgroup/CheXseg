from constants import *
import pandas as pd 
import numpy as np

df = pd.read_csv("/deep/group/CheXpert/CheXpert-v1.0/valid.csv")
df = df.rename(columns={"Lung Opacity": "Airspace Opacity"})
print(np.sum(df[LOCALIZATION_TASKS].values, axis = 0))