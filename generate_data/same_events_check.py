import h5py
import numpy as np
import glob
import os

def compare_files(file1, file2):
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        if f1.keys() != f2.keys():
            print("Keys are different")
            return False

        for key in f1.keys():
            if np.array_equal(f1[key], f2[key]):
                continue
            else:
                print(f"Key {key} is different")
                return False

    return True

def compare_folders(folder1, folder2):
    files1 = sorted(glob.glob(os.path.join(folder1, "*.hdf5")))
    files2 = sorted(glob.glob(os.path.join(folder2, "*.hdf5")))

    nums = np.arange(0, len(files2))
        
    if len(files1) != len(files2):
        print("Number of files is different")
        return False

    for file1, file2 in zip(files1, files2):
        if not compare_files(file1, file2):
            print(f"Files {file1} and {file2} are different")
            return False

    return True

folder1 = "/data/tim/ecddp/dataset/abandonedfactory/Hard/P009/event_left_copy"
folder2 = "/data/tim/ecddp/dataset/abandonedfactory/Hard/P009/event_left"
folder1 = "/data/tim/datasets/E-TartanAir/abandonedfactory/Easy/P000/event_left_chunked_new"
folder2 = "/data/tim/datasets/E-TartanAir/abandonedfactory/Easy/P000/event_left_chunked_old"

print(compare_folders(folder1, folder2))