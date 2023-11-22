import os,sys
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.ndimage.morphology as morph
import SimpleITK as sitk
from img_utils import read3DImage, show3SlicesFromNumpy

device = torch.device('cpu')

tumor_slices = {
    "Brats18_TCIA09_451_1_t1_pveseg":(71,93),
    "Brats18_2013_24_1_t1_pveseg":(75,96),
    "Brats18_TCIA09_428_1_t1_pveseg":(73,98),
    "Brats18_TCIA09_620_1_t1_pveseg":(71,94),
    "Brats18_TCIA12_298_1_t1_pveseg":(73,97),
    "Brats18_TCIA10_442_1_t1_pveseg":(67,91),
    "Brats18_TCIA13_624_1_t1_pveseg":(68,98),
    "Brats18_TCIA10_175_1_t1_pveseg":(77,98),
    "Brats18_TCIA10_152_1_t1_pveseg":(73,94),
    "Brats18_TCIA10_276_1_t1_pveseg":(72,95),
    "Brats18_TCIA09_462_1_t1_pveseg":(72,95)

}

class PairedMaskDataset(Dataset):
    def __init__(self, root, data_path='', mode='train',listnameA='data_list_deformed.txt', listnameB='data_list_undeformed.txt', listnameC='data_list_test.txt', edges=False):
        # listA for deformed masks
        # listB for corresponding undeformed masks
        # listC for test 11 deformed masks from another dataset BraTS 2018
        self.mode = mode
        if self.mode == 'train':# Supervised way, providing deformed-undeformed pair
            self.files_def=open(os.path.join(root,  mode + '/'+listnameA)).readlines()
            self.files_und = open(os.path.join(root,  mode + '/'+listnameB)).readlines()
        else:# input deformed to get undeformed
            self.files_def=open(os.path.join(root,  mode + '/'+listnameC)).readlines()
        self.edges=edges
        self.data_path = data_path

    def enhanceImageQuality(self, img_arr, def_=True):
        img_arr = img_arr.astype('uint8')
        ventr_arr = np.zeros_like(img_arr)
        ventr_arr[img_arr==1] = 1 #int32 now
        ventr_arr = ventr_arr.astype('uint8')
        brain_tissue = np.zeros_like(img_arr)
        brain_tissue[img_arr > 0] = 1
        brain_tissue = brain_tissue.astype('uint8')
        if def_:
            tumor_arr = np.zeros_like(img_arr)
            tumor_arr[img_arr==4]=1

        for i in range(brain_tissue.shape[0]):
            brain_mask_eroded = morph.binary_erosion(brain_tissue[i,:,:].astype('uint8'),iterations=20) # erosion to remove border csf
            if def_:
                tumor_mask = morph.binary_dilation(tumor_arr[i,:,:],iterations=5) # dilation to remove the ring around tumor
                ventr_arr[i,:,:] = ventr_arr[i,:,:] * brain_mask_eroded*np.logical_not(tumor_mask)
            ventr_arr[i,:,:] = ventr_arr[i,:,:] * brain_mask_eroded
            # ventr_arr[i,:,:] = morph.binary_opening(ventr_arr[i,:,:],structure=np.ones((1,1)))
        ventr_arr = ventr_arr.astype('uint8')
        img_new = np.zeros_like(img_arr)
        img_new[ventr_arr==1]=1 #ventricle
        img_new[(img_arr!=0) & (ventr_arr!=1) & (img_arr!=4)]=2 #grey+white
        img_new[img_arr==4]=4 #tumor
        return img_new

    def BraTS2Normal(self,img_arr):
        '''
        Transfer BraTS mask img to normal image.
        1. label value transfer
        2. retain only tumor layers
        3. central crop?
        '''
        img_new = np.zeros_like(img_arr)
        img_new[(img_arr==1)|(img_arr==2)|(img_arr==3)]=4
        img_new[img_arr==4]=2 #grey matter
        img_new[img_arr==5]=3 #white matter
        img_new[img_arr==6]=1 #csf
        img_new = img_new[list(set(np.where(img_new==4)[0])),:,:] #contains only layers with tumor
        return img_new

    def GenerateTumorImg(self,img_arr,slice_selected=None):
        '''
        generate tumor file to paste.
        '''
        if not slice_selected: #if not specified slices.
            img_new = img_arr[list(set(np.where((img_arr>0) & (img_arr<4))[0])),:,:]
        else:
            img_new = img_arr[slice_selected[0]:slice_selected[1],:,:]
        img_new[img_new>=4]=0
        return img_new
    
    def crop_and_resize(self,img_arr,mode='nearest'):
        '''
        Crop and resize img to the specified size.
        '''
        img_arr = torch.from_numpy(img_arr).to(torch.float32)
        img_arr = img_arr.unsqueeze(0).unsqueeze(0)
        out_d = 24
        out_h = 180
        out_w = 180
        new_d = torch.linspace(-1, 1, out_d)
        new_h = torch.linspace(-1, 1, out_h)
        new_w = torch.linspace(-1, 1, out_w)
        mesh_z,mesh_y,mesh_x = torch.meshgrid(new_d,new_h,new_w)
        grid = torch.cat((mesh_x.unsqueeze(3),mesh_y.unsqueeze(3),mesh_z.unsqueeze(3)),dim=3)#equivalent to torch.stack
        grid = grid.unsqueeze(0)
        img_out_arr = F.grid_sample(img_arr,grid,mode=mode)
        img_out_arr = img_out_arr.squeeze(0).squeeze(0).cpu().numpy()
        return img_out_arr

    def __getitem__(self, index):
        segment_class = 3 # TODO: fix HardCoding. merged WM, GM to 2 in enhanceImageQuality
        if self.mode=='train':
            eta=0.000001
            image_def_filename=self.files_def[index % len(self.files_def)].rstrip()
            print(f'{self.data_path}{image_def_filename}')
            image_def = read3DImage(f'{self.data_path}{image_def_filename}')
            image_def = self.enhanceImageQuality(image_def,True)
            image_def_c0 = image_def == 0 # background
            image_def_c1 = image_def == 1 # ventricle
            image_def_c2 = image_def == 2 # gray+white
            tumor_c2 = image_def == 4 # tumor
            image_def_c2 = tumor_c2 + image_def_c2  # gray+white+tumor

            image_def_c0 = morph.distance_transform_edt(image_def_c0)
            image_def_c0 /=np.max(image_def_c0) +eta
            image_def_c1 = morph.distance_transform_edt(image_def_c1)
            image_def_c1 /= np.max(image_def_c1)+eta
            image_def_c2 = morph.distance_transform_edt(image_def_c2)
            image_def_c2 /= np.max(image_def_c2)+eta

            image_und_filename = self.files_und[index % len(self.files_und)].rstrip()
            image_und = read3DImage(f'{self.data_path}{image_und_filename}')
            image_und = self.enhanceImageQuality(image_und, False)

            image_und_c0 = image_und == 0
            image_und_c0 = np.logical_and(image_und_c0, np.logical_not(tumor_c2))
            image_und_c1 = image_und == 1
            image_und_c1=np.logical_and(image_und_c1, np.logical_not(tumor_c2))
            image_und_c2 = image_und == 2
            image_und_c2=np.logical_or(image_und_c2, tumor_c2)

            image_und_c0 = morph.distance_transform_edt(image_und_c0)
            image_und_c0 /= np.max(image_und_c0) + eta
            image_und_c1 = morph.distance_transform_edt(image_und_c1)
            image_und_c1 /= np.max(image_und_c1) + eta
            image_und_c2 = morph.distance_transform_edt(image_und_c2)
            image_und_c2 /= np.max(image_und_c2) + eta
            if self.edges:
                sketch_und_filename = image_und_filename.replace('_segm.png','_sketch.png')
                sketch_und = read3DImage(sketch_und_filename)
                sketch_und = morph.distance_transform_edt(1-sketch_und)
                sketch_und /= np.max(sketch_und) + eta
                sketch_def_filename = image_def_filename.replace('_segm.png', '_sketch.png')
                sketch_def = read3DImage(sketch_def_filename)
                sketch_def = morph.distance_transform_edt(1-sketch_def)
                sketch_def /= np.max(sketch_def) + eta
                image_und_channels = np.stack([image_und_c0, image_und_c1, image_und_c2,sketch_und])
                image_def_channels = np.stack([image_def_c0, image_def_c1, image_def_c2,sketch_def])
            else:
                image_und_channels = np.stack([image_und_c0, image_und_c1, image_und_c2])
                image_def_channels = np.stack([image_def_c0, image_def_c1, image_def_c2])
            image_und_channels= torch.tensor(image_und_channels).float()
            image_und_channels = F.interpolate(image_und_channels.unsqueeze(0), size=(24,180,180), mode="trilinear").squeeze(0)# [batch,channel,x,y,z]
            image_def_channels = torch.tensor(image_def_channels).float()
            image_def_channels = F.interpolate(image_def_channels.unsqueeze(0), size=(24,180,180), mode="trilinear").squeeze(0)
            tumor=tumor_c2
            tumor=torch.tensor(tumor).float()
            tumor=F.interpolate(tumor.unsqueeze(0).unsqueeze(0), size=(24,180,180), mode="nearest").squeeze(0)
            mask=torch.zeros_like(tumor.squeeze(0))
            mask[image_def_channels[0,:,:]==0]=1
            # print(f"PairedMaskDataset: def_image shape = {image_def_channels.float().size()}, und_image shape = {image_und_channels.float().size()}, "
            #       f"mask.unsqueeze(0) shape = {mask.unsqueeze(0).float().size()}, tumor shape = {tumor.float().size()}")
            return {'image_def': image_def_channels.float(), 'image_und': image_und_channels.float(),'mask':mask.unsqueeze(0).float(),'tumor':tumor.float()}

        else:#test
            if segment_class==3:
                eta=0.000001
                image_def_filename=self.files_def[index % len(self.files_def)].rstrip()
                image_def = read3DImage(f'{self.data_path}{image_def_filename}')
                # print(image_def_filename)
                # show3SlicesFromNumpy(image_def, 'Show BraTS')
                #if the image is BraTS:
                # slice_selected = tumor_slices[os.path.basename(f'{self.data_path}{image_def_filename}').split(".")[0]] # TODO: why these two steps are needed?
                # self.generated_tumor = self.crop_and_resize(self.GenerateTumorImg(image_def, slice_selected))
                image_def = self.BraTS2Normal(image_def)
                image_def = self.enhanceImageQuality(image_def)
                # show3SlicesFromNumpy(image_def, 'Show BraTS after adj')
                image_def_c0 = image_def == 0
                image_def_c1 = image_def == 1
                image_def_c2 = image_def == 2
                tumor_c2 = image_def == 4
                image_def_c2 = tumor_c2 + image_def_c2

                image_def_c0 = morph.distance_transform_edt(image_def_c0)
                image_def_c0/=np.max(image_def_c0) +eta
                image_def_c1 = morph.distance_transform_edt(image_def_c1)
                image_def_c1 /= np.max(image_def_c1)+eta
                image_def_c2 = morph.distance_transform_edt(image_def_c2)
                image_def_c2 /= np.max(image_def_c2)+eta
                image_def_channels = np.stack([image_def_c0, image_def_c1, image_def_c2])
                image_def_channels = torch.tensor(image_def_channels).float()
                image_def_channels = F.interpolate(image_def_channels.unsqueeze(0), size=(24,180,180), mode="trilinear").squeeze(0)
                # print(torch.sum(image_def_channels,0)[50,:,:])
                tumor=tumor_c2
                tumor=torch.tensor(tumor).float()
                tumor=F.interpolate(tumor.unsqueeze(0).unsqueeze(0), size=(24,180,180), mode="nearest").squeeze(0)
                mask=torch.zeros_like(tumor.squeeze(0))
                mask[image_def_channels[0,:,:]==0]=1
                return {'image_def_original':image_def, 'image_def': image_def_channels.float(),'mask':mask.unsqueeze(0).float(),'tumor':tumor.float()} # 'generated_tumor':self.generated_tumor
            
    def __len__(self):
        # return max(self.files_def.__len__(),self.files_und.__len__())
        return self.files_def.__len__()

if __name__ == "__main__":
    print("")
