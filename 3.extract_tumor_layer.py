import os
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
#读取含有肿瘤的层数，并存成tiff格式文件；#jpg,png会有压缩损失，存multi label文件会有问题
maskpath = '/data_dir/Segmented/LGG'
maskfile = glob(f'{maskpath}/*t1_pveseg.nii.gz') #need to specify
imgpath = '/data_dir/MICCAI_BraTS_2018_Data_Training/LGG'
imgfile = glob(f'{imgpath}/**/*_t1.nii.gz', recursive=True)
os.makedirs('/data_dir/png_file/tmp/tmp_img_T1_tiff',exist_ok=True)
os.makedirs('/data_dir/png_file/tmp/tmp_label_T1_tiff',exist_ok=True)
os.makedirs('/data_dir/png_file/tmp/tmp_img_tiff',exist_ok=True)
os.makedirs('/data_dir/png_file/tmp/tmp_label_tiff',exist_ok=True)
os.makedirs('/data_dir/png_file/tmp/tmp_img_png_T1',exist_ok=True)
os.makedirs('/data_dir/png_file/tmp/tmp_label_png_T1',exist_ok=True)
# storepath = '/data_dir/png_file/tmp/tmp_img_tiff' #train_img,train_label
storepath = '/data_dir/png_file/tmp/tmp_label_png_T1'



for img,mask in tqdm(zip(sorted(imgfile),sorted(maskfile))):
    mask_img = sitk.ReadImage(mask)
    # mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    img_img = sitk.ReadImage(img)
    # img_img = sitk.Cast(img_img, sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(img_img)
    for i in range(mask_img.GetSize()[2]):
        tmp_mask_arr = mask_arr[i,:,:]
        tmp_img_arr = img_arr[i,:,:]
        if (1 in tmp_mask_arr) or (2 in tmp_mask_arr) or (3 in tmp_mask_arr): #if contains tumor
            png_img = sitk.GetImageFromArray(tmp_mask_arr)
            # png_img = sitk.RescaleIntensity(png_img)
            png_img = sitk.Cast(png_img,sitk.sitkUInt8)
            name = img.split('/')[-1].split('.')[0]
            # im = Image.fromarray(tmp_img_arr)
            # im.convert('L').save(f'{storepath}/{name}_{i}.jpg')
            # im.save(f'{storepath}/{name}_{i}.jpg')
            # sitk.WriteImage(png_img,f'{storepath}/{name}_{i}.tiff')
            sitk.WriteImage(png_img,f'{storepath}/{name}_{i}.png')






