
# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
import pickle as pkl
from efm_datasets.utils.decorators import multi_write
from efm_datasets.utils.types import is_tensor
import torchvision.transforms as transforms
import numpy as np
import cv2 


def create_folder(filename):
    """Create a new folder from filename if it doesn't exist"""
    # Create folder if it doesn't exist
    if '/' in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)


def write_pickle(filename, data):
    """Write pickle to filename"""
    create_folder(filename)
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    pkl.dump(data, open(filename, 'wb'))


def write_npz(filename, data):
    """Write npz to filename"""
    # Create folder if it doesn't exist
    create_folder(filename)
    # Save npz
    np.savez_compressed(filename, **data)


@multi_write
def write_depth(filename, depth, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    depth : np.array or torch.Tensor
        Depth map [H,W]
    intrinsics : np.array
        Optional camera intrinsics matrix [3,3]
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If depth is a tensor
    if is_tensor(depth):
        depth = depth.detach().squeeze().cpu().numpy()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, depth=depth, intrinsics=intrinsics)
    # If we are saving as a .png
    elif filename.endswith('.png'):
        depth = transforms.ToPILImage()((depth * 256).astype(np.int32))
        depth.save(filename)
    # Something is wrong
    else:
        raise NotImplementedError('Depth filename not valid.')


def write_empty_txt(filename, folder):
    """Write an empty txt file to filename"""
    os.makedirs(folder, exist_ok=True)
    with open(f'{folder}/{filename.replace("/", "|")}.txt', 'w') as f: pass
    
@multi_write    
def write_image(filename, image):
    """    Write an image to file

    Parameters
    ----------
    filename : String
        File where image will be saved
    image : np.Array [H,W,3]
        RGB image
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If image is a tensor
    if is_tensor(image):
        if len(image.shape) == 4:
            image = image[0]
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # If image is a numpy array
    elif isinstance(image, np.ndarray):
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # Otherwise, assume it's a PIL image
    else:
        image.save(filename)
    # return np.clip(image, 0, 255).astype(np.uint8)
