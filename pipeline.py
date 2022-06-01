import argparse
import os,sys
import numpy as np
import SimpleITK as sitk
from glob import glob 
from tqdm import tqdm


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Tumor Generation')
    parser.add_argument('--def_datapath', type=str, default='./Lists', help='deformed data of the dataset')
    parser.add_argument('--normal_datapath', type=str, default='./Lists', help='data to be deformed')
    parser.add_argument('--slices',type=int, default=1,help='slices selected to display')
    parser.add_argument('--device', type=int, default=0)
    
    device = torch.device('cuda', opt.device)
    opt = parser.parse_args()


