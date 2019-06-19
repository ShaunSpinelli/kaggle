from pathlib import Path
import argparse
import numpy as np
from shutil import move


"""Splits image data sets into train and validation"""

def get_files(data_dir):
    files = {}
    for folder in data_dir.iterdir():
        files[folder] = [i for i in folder.iterdir()]
    return files

def split(files, size=0.8):
    np.random.shuffle(files)
    split = int(len(files) * size)
    train = files [:split]
    valid = files [split:]
    return train, valid

def move_files(files, new):
    print("Moving Files")
    new.mkdir(exist_ok=True, parents=True)
    for file in files:
        move(file, new/file.name)



def split_classes(files):
    for key,value in files.items():
        train,valid = split(value)
        move_files(valid,Path("valid")/key.name )
        move_files(train ,Path("train")/key.name )


def main(data_dir):
    """splits up images data set
    args:
      data_dir type:Path         
    """
    files = get_files(data_dir)
    split_classes(files)
    # print(files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Splits image data sets into train and validation")
    # parser.add_argument("data_dir", type=Path, help="")
    # parser.add_argument("data_dir", type=Path, help="")
    args = parser.parse_args()


    main(args.data_dir)
