import os
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

filepaths = glob(f'/data_dir/MICCAI_BraTS_2018_Data_Training/LGG/**/*_t1.nii.gz', recursive=True)

for filepath in tqdm(filepaths):
    os.system(f'/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o {filepath} {filepath} ')