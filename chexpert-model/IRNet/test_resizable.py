import h5py
import sys
import os
sys.path.append(os.path.abspath("../"))

from constants import *

ROOT_DIR = CHEXPERT_PARENT_TRAIN_CAMS_DIR / 'hdf5_files'
f2 = h5py.File(ROOT_DIR / "train_set_ten_percent_0")
f3 = h5py.File(ROOT_DIR / "train_set_ten_percent_100")


with h5py.File(ROOT_DIR / "resizable", "a") as f:
    #dset = f.create_dataset("cams", data=f2["cxr_cams_ten_percent"][:], maxshape=(None, None, None, None, None))
    dset = f['cams']
    # dset.resize(4000, axis=0)
    dset[32:3232] = f3["cxr_cams_ten_percent"][:]
    print(dset[34])