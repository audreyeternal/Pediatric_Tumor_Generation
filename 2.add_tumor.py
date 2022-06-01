import os
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
#1.remove the tumor part, and set it to 0, and save it;
#2.use fast segmentation, and remove it;
# tumor_img = sitk.ReadImage('/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_2013_0_1/Brats18_2013_0_1_seg.nii.gz')
# tumor_arr = sitk.GetArrayFromImage(tumor_img)
# img = sitk.ReadImage('/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_2013_0_1/Brats18_2013_0_1_t1ce.nii.gz')
# arr = sitk.GetArrayFromImage(img)

# input_path_undeformed = '/host_project/Brats18_2013_0_1_t2.nii.gz'
# input_mask = '/host_project/Brats18_2013_0_1_seg.nii.gz'
# input_img = sitk.ReadImage(input_path_undeformed)
# mask_img = sitk.ReadImage(input_mask) #test
# input_arr = sitk.GetArrayFromImage(input_img)
# mask_arr = sitk.GetArrayFromImage(mask_img)
# input_arr[mask_arr!=0]=0
# input_img = sitk.GetImageFromArray(input_arr)
# sitk.WriteImage(input_img,'tmp_img.nii')
# os.system(f'/usr/local/fsl/bin/fast -t 2 -n 3 -H 0.1 -I 4 -l 20.0 -o tmp_img tmp_img')

#read the whole folder and do the segmentation:
#TODO: 1. 将mask文件合二为一:
morepaths_T1 = glob(f'/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/**/*_t1_seg.nii.gz', recursive=True)
morepaths_T2 = glob(f'/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/**/*_t2_seg.nii.gz', recursive=True)
imgpaths = glob(f'/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/**/*_t1_pveseg.nii.gz', recursive=True)
tumorpaths = glob(f'/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/**/*_seg.nii.gz', recursive=True)
tumorpaths = list(set(tumorpaths)-set(morepaths_T1))#求集合差集；
tumorpaths = list(set(tumorpaths)-set(morepaths_T2))#求集合差集；
print(tumorpaths)
for img,tumor in tqdm(list(zip(sorted(imgpaths),sorted(tumorpaths)))):
    img_img = sitk.ReadImage(img)
    img_arr = sitk.GetArrayFromImage(img_img)
    img_arr[img_arr!=0] += 3
    #specific for T1 FAST segmentation:
    img_arr[img_arr==4]=7
    img_arr[img_arr==5]=4
    img_arr[img_arr==6]=5
    img_arr[img_arr==7]=6
    tumor_img = sitk.ReadImage(tumor)
    tumor_arr = sitk.GetArrayFromImage(tumor_img)
    tumor_arr[tumor_arr==4]=3 #NCR区域为4，要改为3；
    img_arr[tumor_arr!=0] = tumor_arr[tumor_arr!=0]
    new_img = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(new_img,f'/data_dir/Segmented/LGG/{os.path.basename(img)}')


# for filepath in tqdm(filepaths):
#     os.system(f'/usr/local/fsl/bin/fast -t 2 -n 3 -H 0.1 -I 4 -l 20.0 -o {filepath} {filepath} ')

