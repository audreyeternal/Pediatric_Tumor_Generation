import os
from glob import glob
import cv2 as cv
from PIL import Image, ImageOps
from tqdm.contrib import tenumerate
import SimpleITK as sitk
import numpy as np


def padding(img_arr,pad_size=256):
    size = img_arr.shape
    padding_width = (pad_size-size[0])//2
    padding_height = (pad_size-size[1])//2
    img_arr = np.pad(img_arr,((padding_width,padding_width),(padding_height,padding_height)),mode='constant',constant_values=0)
    return img_arr

image_store_path = '/host_project/Pytorch-Medical-Segmentation/data/augment_3080/train/image'
mask_store_path = '/host_project/Pytorch-Medical-Segmentation/data/augment_3080/train/label'
os.makedirs(image_store_path,exist_ok=True)
os.makedirs(mask_store_path,exist_ok=True)

os.system(f'cp -r /host_project/SPADE/results/LGG_greyscale_256_loadsize_256_crop_size_256_mode_T1_1/test_latest/images/synthesized_image/* {image_store_path}')

img_path = glob('/host_project/Pytorch-Medical-Segmentation/data/augment_3080/train/image/*')

for img in img_path:
    img_file = Image.open(img).convert('L')
    img_file.save(img)



os.system(f'cp -r /data_dir/png_file/train_img_png_T1/* {image_store_path}')
os.system(f'cp -r /data_dir/png_file/train_label_png_T1/* {mask_store_path}')
os.system(f'cp -r /host_project/TumorMassEffect/images/val_img/* {mask_store_path}')

img_path = glob('/host_project/Pytorch-Medical-Segmentation/data/augment_3080/train/image/*')
mask_path = glob("/host_project/Pytorch-Medical-Segmentation/data/augment_3080/train/label/*")


for i,(img,mask) in enumerate(zip(sorted(img_path),sorted(mask_path))):
    img_file = sitk.GetArrayFromImage(sitk.ReadImage(img))
    mask_file = sitk.GetArrayFromImage(sitk.ReadImage(mask))

    # print(img_file.size,mask_file.size)
    if img_file.shape[0]<256:
        img_file = padding(img_file, 256)
        sitk.WriteImage(sitk.GetImageFromArray(img_file),img)

    if mask_file.shape[0]<256:
        mask_file = padding(mask_file, 256)
        sitk.WriteImage(sitk.GetImageFromArray(mask_file),mask)
        


    