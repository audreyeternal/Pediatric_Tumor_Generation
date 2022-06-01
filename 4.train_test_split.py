import os
import numpy as np 
from glob import glob
from tqdm import tqdm

valPr = 0.1

os.makedirs('/data_dir/png_file/train_img_png_T1',exist_ok=True)
os.makedirs('/data_dir/png_file/train_label_png_T1',exist_ok=True)
os.makedirs('/data_dir/png_file/val_img_png_T1',exist_ok=True)
os.makedirs('/data_dir/png_file/val_label_png_T1',exist_ok=True)

imgpaths = glob(f'/data_dir/png_file/tmp/tmp_img_png_T1/*')
maskpaths = glob(f'/data_dir/png_file/tmp/tmp_label_png_T1/*')
train_img_store_path = '/data_dir/png_file/train_img_png_T1'
train_mask_store_path = '/data_dir/png_file/train_label_png_T1'
valid_img_store_path = '/data_dir/png_file/val_img_png_T1'
valid_mask_store_path = '/data_dir/png_file/val_label_png_T1'
num_train=len(imgpaths)
indices=list(range(num_train))
split=int(np.floor(valPr*num_train))
np.random.seed(0) #固定住
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
imgpaths = np.array(sorted(imgpaths))
maskpaths = np.array(sorted(maskpaths))
train_img_paths = imgpaths[train_idx]
valid_img_paths = imgpaths[valid_idx]
train_mask_paths = maskpaths[train_idx]
valid_mask_paths = maskpaths[valid_idx]

# print(imgpaths[:15],maskpaths[:15])
#move the file:
for i,path in enumerate(train_img_paths):
    os.system(f'cp {path} {train_img_store_path}')

for path in valid_img_paths:
    os.system(f'cp {path} {valid_img_store_path}')

for path in train_mask_paths:
    os.system(f'cp {path} {train_mask_store_path}')

for path in valid_mask_paths:
    os.system(f'cp {path} {valid_mask_store_path}') 