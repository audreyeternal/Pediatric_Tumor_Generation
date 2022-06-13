import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tenumerate
import cv2
from glob import glob
from random import choice,sample

case_list = ["002","003","004","010","011","013","019","021","023","024","025","026","028","029"]
slice_list = list(range(2,22))
tumor_list = list(range(0,11))

def concat_vh(list_2d):
    """
    concat 2d images vertically or horizontally
    """
    return cv2.vconcat([cv2.hconcat(list_h) 
                        for list_h in list_2d])

def concat_h(list_h):
    return cv2.hconcat(list_h)

def change_color(img):
    RED = np.array([0,0,128]).astype(np.uint8)
    BLACK = np.array([0,0,0]).astype(np.uint8)
    WM = np.array([128,128,0]).astype(np.uint8)
    WM_after = np.array([240,255,240]).astype(np.uint8)
    CSF = np.array([128,128,128]).astype(np.uint8)
    CSF_after = np.array([238,134,28]).astype(np.uint8)
    GM = np.array([128,0,128]).astype(np.uint8)
    GM_after = np.array([205,182,159]).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i,j]==RED).all():
                img[i,j]=BLACK
            if (img[i,j]==WM).all():
                img[i,j]=WM_after
            if (img[i,j]==CSF).all():
                img[i,j]=CSF_after
            if (img[i,j]==GM).all():
                img[i,j]=GM_after
    return img


def plot_images(case_num,slice_num,tumor_num):
    origin_img = cv2.imread("/host_project/TumorMassEffect/images/origin_openneuro_img/sub-pixar" + case_num + "_anat_sub-pixar" + case_num + f"_T1w_sliced_{slice_num}.png",0)
    origin_img = cv2.resize(origin_img, (256,256), interpolation = cv2.INTER_AREA)
    deformed_mask = cv2.imread("/host_project/SPADE/results/LGG_greyscale_256_loadsize_256_crop_size_256_mode_T1_1/test_latest/images/input_label/sub-pixar" + case_num + "_anat_sub-pixar" + case_num + f"_T1w_sliced_{slice_num}_tumor_{tumor_num}.png")
    deformed_mask = change_color(deformed_mask)
    deformed_mask = cv2.resize(deformed_mask, (256,256), interpolation = cv2.INTER_AREA)
    deformed_T1 = cv2.imread(f"/host_project/SPADE/results/LGG_greyscale_256_loadsize_256_crop_size_256_mode_T1_1/test_latest/images/synthesized_image/sub-pixar" + case_num + "_anat_sub-pixar" + case_num + f"_T1w_sliced_{slice_num}_tumor_{tumor_num}.png",0)
    defomred_T2 = cv2.imread(f"/host_project/SPADE/results/LGG_greyscale_256_loadsize_256_crop_size_256_mode_T2_1/test_latest/images/synthesized_image/sub-pixar" + case_num + "_anat_sub-pixar" + case_num + f"_T1w_sliced_{slice_num}_tumor_{tumor_num}.png",0)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2RGB)
    deformed_T1 = cv2.cvtColor(deformed_T1, cv2.COLOR_GRAY2RGB)
    defomred_T2 = cv2.cvtColor(defomred_T2, cv2.COLOR_GRAY2RGB)
    return concat_h([origin_img,deformed_mask,deformed_T1,defomred_T2])

    
if __name__ == "__main__":

    # print(case_num,slice_num,tumor_num)
    img_list = []
    case_random_list = sample(case_list,5)
    for case in case_random_list:
        print(case)
        slice_num = choice(slice_list)
        tumor_num = choice(tumor_list)
        img_list.append(plot_images(case,slice_num,tumor_num))
    img = cv2.vconcat(img_list)
    cv2.imwrite("/host_project/synthesized_images.png",img)
    

    

