from __future__ import print_function
import argparse
import itertools
import os,sys
from math import log10
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_UNet import ProbabilisticUnet, l2_regularisation
from dataset import PairedMaskDataset
from net_utils import calc_class_weights,warp_image, warp_image_diffeomorphic, diffusion_regularise_displacement, GaussianLayer, sparsity_regularise_displacement,inverse_warp_image_diffeomorphic
from torch.nn import functional as F
from img_utils import save3DImage
from multiprocessing import Process, freeze_support
import pickle
from torchsummary import summary
from PIL import Image
from skimage import io 
import shutil 
from glob import glob

def read_image(path,front=0,back=0):
    img = sitk.ReadImage(path)
    size = img.GetSize()[2]
    print(list(range(front,(size-back))))
    img = img[:,:,front:(size-back-1)]
    return img

def pad(img_arr,pad_size=240):
    size = img_arr.shape
    padding_width = (pad_size-size[0])//2
    padding_height = (pad_size-size[1])//2
    img_arr = np.pad(img_arr,((padding_width,padding_width),(padding_height,padding_height)),mode='constant',constant_values=0)
    return img_arr

def generate_spade_image(img_arr,slice_num=12,if_mask=True):
    '''
    select a slice, and pad to SPADE size.
    '''
    img_arr_slice = img_arr[slice_num,:,:].copy()
    #padding to 240:
    img_arr_slice = pad(img_arr_slice,pad_size=240)
    img_arr_slice = sitk.GetImageFromArray(img_arr_slice)
    if not if_mask: #if original image:
        img_arr_slice = sitk.RescaleIntensity(img_arr_slice)
    img_arr_slice = sitk.Cast(img_arr_slice,sitk.sitkUInt8)
    return img_arr_slice

def resize(origin_img,size=[24,180,180],mode='bilinear',flip=False):
    '''
    resize img to specified size. Sometimes we need to flip the image in y direction after resize.
    input: origin_img, sitkImage
    output: img_out_arr, torch.tensor
    '''
    img_arr = sitk.GetArrayFromImage(origin_img).astype(np.float32)
    img_arr = torch.tensor(img_arr).to(dtype=torch.float32)
    img_arr = img_arr.unsqueeze(0).unsqueeze(0)
    out_d,out_h,out_w = size
    new_d = torch.linspace(-1, 1, out_d)
    new_h = torch.linspace(-1, 1, out_h)
    new_w = torch.linspace(-1, 1, out_w)
    mesh_z,mesh_y,mesh_x = torch.meshgrid(new_d,new_h,new_w)
    grid = torch.cat((mesh_x.unsqueeze(3),mesh_y.unsqueeze(3),mesh_z.unsqueeze(3)),dim=3)#等价于torch.stack
    grid = grid.unsqueeze(0)
    img_out_arr = F.grid_sample(img_arr,grid,mode=mode)
    img_out_arr = img_out_arr.squeeze(0).squeeze(0)
    if flip:
        img_out_arr = torch.flip(img_out_arr,[1,2])
    return img_out_arr

def segment(image_wrapped_sitk_arr):
    image_wrapped_sitk_arr[(image_wrapped_sitk_arr>=0)&(image_wrapped_sitk_arr<0.2)]=0
    image_wrapped_sitk_arr[(image_wrapped_sitk_arr>=0.2)&(image_wrapped_sitk_arr<1.8)]=6 #csf
    image_wrapped_sitk_arr[(image_wrapped_sitk_arr>=1.8)&(image_wrapped_sitk_arr<2.6)]=4 #grey
    image_wrapped_sitk_arr[(image_wrapped_sitk_arr>=2.6)&(image_wrapped_sitk_arr<4)]=5 #white
    return image_wrapped_sitk_arr

# Training settings
if __name__== '__main__':
    
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--dataroot', type=str, default='./Lists', help='root directory of the dataset')
    parser.add_argument('--resultsroot', type=str, default='./Results', help='root directory of the dataset')
    parser.add_argument('--experiment_type', type=str, default='TumorDeformations', help='root directory of the dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--valPr', type=int, default=0.05)
    parser.add_argument('--startRegWeight', type=int, default=0.002)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--traintype',type=str, default='def',help="Types:[normal, adv, cycle, def]")
    opt = parser.parse_args()

    # torch.manual_seed(20)
    device = torch.device('cuda', opt.device)
    print('===> Loading datasets')
    inputDir = os.path.join(opt.dataroot,opt.experiment_type)
    outputDir=os.path.join(opt.resultsroot,opt.experiment_type)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    valOutDir=os.path.join(outputDir,'Val')
    if not os.path.exists(valOutDir):
        os.makedirs(valOutDir)

    train_set = PairedMaskDataset(inputDir)
    val_set=PairedMaskDataset(inputDir)
    test_set = PairedMaskDataset(inputDir,mode='test')
    
    num_train=len(train_set)
    indices=list(range(num_train))
    split=int(np.floor(opt.valPr*num_train))
    np.random.seed(32)
    np.random.shuffle(indices)
    # print(f'the number of indices is {len(indices)}')
    print(f'the split is {split}')
    train_idx, valid_idx = indices[split:], indices[:split]#2022-2-27. train_test split
    print(f'train_idx:{train_idx},valid_idx:{valid_idx}')# 2022-03-01 valid_idx为空(因为总数据为1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # print(f'train_sample:{train_sampler}')
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=opt.batchSize, sampler=train_sampler)
    validation_data_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, sampler=valid_sampler)
    test_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1)


    print('===> Building model')





    with torch.cuda.device(opt.device):
        netG=ProbabilisticUnet(input_channels=4,dim=3, num_filters=[32,64,128], latent_dim=15, no_convs_fcomb=3, beta=1)#segment class 4.
        criterionGAN = nn.CrossEntropyLoss() #2022-03-01 好像没有用到
        criterionL1=nn.L1Loss() #2022-03-01 好像没有用到
        optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
        
        
        
        print('---------- Networks initialized -------------')
        
        
        
        netG=netG.to(device)
        Gauss = GaussianLayer(sigma=3).to(device)
        criterionGAN = criterionGAN.to(device)
        criterionL1=criterionL1.to(device)

    def validate(epoch):
        netG.eval()
        with torch.no_grad():
            for i,batch in enumerate(validation_data_loader):
                if i<2:
                    images_und = batch['image_und'].to(device)
                    images_def = batch['image_def'].to(device)
                    mask=batch['mask'].to(device)
                    tumor=batch['tumor'].to(device)
                    netG(torch.cat((images_def,tumor),dim=1),None, training=False)
                    #==============validate时无法使用elbo损失，因为posterior encoder不工作；============== 2022-03-13
                    displ=netG.sample()
                    displ = displ.permute(0, 2, 3,4, 1)
                    #computed_displ=displ
                    prediction, computed_displ=warp_image_diffeomorphic(images_def, displ, mode='bilinear', ret_displ=True)
                    # print(f'=============the type of the computed_displ is {type(computed_displ)}======================')
                    #prediction=warp_image(images_def,displ,mode='bilinear')
                    lossGAN = criterionL1(prediction, images_und)
                    prediction=torch.argmax(prediction[:,:,:,:,:],dim=1)
                    images_und=torch.argmax(images_und[:,:,:,:,:],dim=1)
                    images_def = torch.argmax(images_def[:,:,:,:,:], dim=1)
                    save3DImage(prediction.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir+"/Val/prediction_"+str(i)+".nii")
                    save3DImage(images_und.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir + "/Val/realund_" + str(i) +".nii")
                    save3DImage(images_def.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir + "/Val/realdef_" + str(i) +".nii")

                    # fig = plt.figure()
                    # x, y = np.meshgrid(np.arange(0, images_def.data.cpu().numpy()[0, :, :].shape[0]),
                    #                 np.arange(0, images_def.data.cpu().numpy()[0, :, :].shape[1]))
                    # x_plt = plt.imshow(
                    #     computed_displ.data.cpu().numpy()[0,15, :, :, 0] ** 2 + computed_displ.data.cpu().numpy()[0, 15,:, :, 1] ** 2)
                    # plt.quiver(x[::5,::5], y[::5,::5], computed_displ.data.cpu().numpy()[0, 15,:, :, 0][::5,::5], computed_displ.data.cpu().numpy()[0, 15,:, :, 1][::5,::5])
                    # cbar = fig.colorbar(x_plt)
                    # cbar.minorticks_on()
                    # fig.savefig(outputDir + "/Val/displ_" + str(i) + ".png")
            print("===> Valid Loss: "+str(lossGAN.item()))

    def train_def(epoch,reg_weight):
        netG.train()
        tumor_reg_weight=100#搞清楚这几个parameter在文中的具体对应
        reg_loss_mean=0
        elbo_mean=0
        if epoch % 50 == 0 and reg_weight<0.001:
            reg_weight *= 2 #每50个epoch翻倍
            print("Reg weight changed: "+str(reg_weight))
        for iteration, batch in enumerate(training_data_loader, 1):
            images_und = batch['image_und'].to(device)
            # sitk.WriteImage(sitk.GetImageFromArray(images_und.cpu().numpy()),r'images/undeformed_test.nii')
            images_def = batch['image_def'].to(device)
            tumor=batch['tumor'].to(device)
            print(f'the shape of the concat:{torch.cat((images_def,tumor),dim=1).shape}')
            netG(torch.cat((images_def,tumor),dim=1),torch.cat((images_und,tumor),dim=1)) #2022-03-07需要理解一下,dimension是5,故应是拼接在channel上。
            elbo = netG.elbo(images_def, images_und)#2022-03-16 为何是这两个做elbo? 因为在elbo函数里要做reconstruction loss.
            print(f'===========the elbo is {elbo.item()}============')
            displ_reg_loss=sparsity_regularise_displacement(netG.displ) # velocity reg,displ在elbo()中定义
            tumor_cut=(torch.max(tumor)-tumor)
            soft_mask=tumor_reg_weight*Gauss(tumor_cut).detach()
            soft_mask=soft_mask+1 # mehr gewichten
            soft_mask = torch.stack((soft_mask[:,0,...],soft_mask[:,0,...],soft_mask[:,0,...]), dim=4)
            displ_reg_loss=displ_reg_loss*soft_mask
            tumor_reg=diffusion_regularise_displacement(netG.displ)
            loss = -elbo +reg_weight*(tumor_reg.mean()+displ_reg_loss.mean()) #2022-03-01 所有的loss
            reg_loss_mean+=reg_weight*(tumor_reg.sum()+displ_reg_loss.mean()).item()
            elbo_mean+=-elbo.item()
            print("Reg loss: " + str(reg_loss_mean))
            print("Elbo:" + str(elbo_mean))
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            save3DImage(netG.computed_displ.data[0, :, :, :, 1].cpu().numpy(),
                        outputDir + "/Val/displ_" + str(iteration) + ".nii")
            print("===> Epoch[{}]({}/{}): Loss G: {:.6f}".format(
                epoch, iteration, len(training_data_loader), loss.item()))
        return reg_weight, reg_loss_mean/(iteration+1), elbo_mean/(iteration+1)


    def test():
        netG.eval()
        try:
            model = torch.load(f'Results/TumorDeformations/netG_epoch_{opt.nEpochs}.pth')
        except:
            print('no pth file provided!')
        netG.load_state_dict(model)
        # summary(netG,input_size=(4,128,128,24),batch_size=2)

        with torch.no_grad():
            for i,batch in enumerate(test_data_loader):
                images_def = batch['image_def'].to(device)
                tumor=batch['tumor'].to(device)
                generated_tumor = batch['generated_tumor'].squeeze(0)
                # print(f'=====the size of test data is {torch.cat((images_def,tumor),dim=1).shape}=========')
                netG(torch.cat((images_def,tumor),dim=1),None, training=False)
                displ=netG.sample()
                displ = displ.permute(0, 2, 3, 4, 1)
                # print(displ.shape)
                #computed_displ=displ
                prediction, computed_displ=warp_image_diffeomorphic(images_def, displ, mode='bilinear', ret_displ=True)
                # print(f'---------prediction 尺寸:{prediction.shape}--------')#[1, 3, 24, 128, 128]
                # print(computed_displ.shape)#(1,100,100,100,3)
                filepaths = glob('/data_dir/OpenNeuro_children/*/*frac.nii.gz',recursive=True)
                for filepath in filepaths:

                # origin_img_classified_path = r"/host_project/TumorMassEffect/images/sub-pixar019_anat_sub-pixar019_T1w_sliced.pvc.frac.nii.gz"
                    origin_img_classified = read_image(filepath,front=3,back=3) # have value from 0 - 4.
                # origin_img = read_image('/host_project/TumorMassEffect/images/sub-pixar019_anat_sub-pixar019_T1w_sliced.bse.nii.gz',front=3,back=3)
                #intepolate using grid_sample:
                    img_out_arr = resize(origin_img_classified,size=[24,180,180],flip=True)
                # origin_img_download = sitk.GetImageFromArray(resize(origin_img,size=[24,180,180],flip=True).to(dtype=torch.float32).cpu().numpy())
                # sitk.WriteImage(origin_img_download,'images/test.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(img_out_arr),'images/pdt_df/origin.nii.gz')
                    #do the deformation:
                    deform_img_arr = img_out_arr.to(device=device,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                #test:
                # test_arr = deform_img_arr.clone()
                # sitk.WriteImage(generate_spade_image(segment(test_arr.squeeze(0).squeeze(0).cpu().numpy())),'images/test_2.tiff')

                    image_wrapped = inverse_warp_image_diffeomorphic(deform_img_arr,computed_displ,mode='bilinear')
                    image_wrapped_sitk_arr = image_wrapped.squeeze(0).squeeze(0).cpu().numpy()
                
                # #match SPADE label with TumorMassEffect label:
                    image_wrapped_sitk_arr = segment(image_wrapped_sitk_arr)
                    image_wrapped_sitk_arr[generated_tumor!=0] = generated_tumor[generated_tumor!=0]
                # sitk.WriteImage(sitk.GetImageFromArray(image_wrapped_sitk_arr),r'images/pdt_df/openneuro_019_deformed.nii.gz')
                # #generate tiff file for SPADE:
                    for j in range(20):
                        slice_num = j+2
                        slice_for_SPADE_deformed = generate_spade_image(image_wrapped_sitk_arr,slice_num=slice_num)
                        tumor_label = np.zeros_like(sitk.GetArrayFromImage(slice_for_SPADE_deformed))
                        slice_for_SPADE_deformed_arr = sitk.GetArrayFromImage(slice_for_SPADE_deformed)
                        tumor_label[slice_for_SPADE_deformed_arr==1] = 1
                        tumor_label[slice_for_SPADE_deformed_arr==2] = 1
                        tumor_label[slice_for_SPADE_deformed_arr==3] = 1
                        # sitk.WriteImage(slice_for_SPADE_deformed,f'images/val_img/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.tiff')
                        # sitk.WriteImage(slice_for_SPADE_deformed,f'images/val_label/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.tiff')
                        # sitk.WriteImage(sitk.GetImageFromArray(tumor_label),f'images/tumor_label/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.tiff')
                        sitk.WriteImage(slice_for_SPADE_deformed,f'images/val_img/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.png')
                        sitk.WriteImage(slice_for_SPADE_deformed,f'images/val_label/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.png')
                        sitk.WriteImage(sitk.GetImageFromArray(tumor_label),f'images/tumor_label/{os.path.basename(filepath).split(".")[0]}_{slice_num}_tumor_{i}.png')

                # slice_for_SPADE_deformed = generate_spade_image(image_wrapped_sitk_arr,slice_num=12)
                # # slice_for_SPADE_origin = generate_spade_image(resize(origin_img,size=[24,180,180],flip=True).to(dtype=torch.float32).cpu().numpy(),slice_num=12,if_mask=False)
                # sitk.WriteImage(slice_for_SPADE_deformed,'images/val_img/slice_for_SPADE_Brats18_TCIA09_451_openneuro_019.jpg')
                # sitk.WriteImage(slice_for_SPADE_deformed,'images/val_label/slice_for_SPADE_Brats18_TCIA09_451_openneuro_019.jpg')
                # sitk.WriteImage(slice_for_SPADE_origin,'images/origin_openneuro_img/slice_for_SPADE_loadsize_256_cropsize_256_openneuro_019.jpg')
                # test_arr = sitk.GetArrayFromImage(slice_for_SPADE)
                # io.imsave('images/val_img/test.jpg',test_arr.astype(np.uint32))
                # image_wrapped_sitk = sitk.GetImageFromArray(image_wrapped_sitk_arr)
                # # sitk.WriteImage(image_wrapped_sitk,r'images/testImage_IXI024-Guys-0705-T1_pveseg_sliced.nii')
                # sitk.WriteImage(image_wrapped_sitk,r'images/pdt_df/testImage_IXI024-Guys-0705-T1_pveseg_sliced.nii')

    
                


    with torch.cuda.device(opt.device):
        # freeze_support()
        train_losses = []
        val_losses = []
        if opt.mode=="train":
            reg_weight=opt.startRegWeight
            # reg_weight = 0.2
            for epoch in range(1, opt.nEpochs + 1):
                if opt.traintype == 'def':
                    # only the G has changed
                    reg_weight, reg_loss, elbo=train_def(epoch,reg_weight)
                    val_losses.append(reg_loss)
                    train_losses.append(elbo)
                    print(f'the training loss is {elbo}')
                if epoch % 1 == 0:
                    torch.save(netG.state_dict(), outputDir + '/netG_epoch_' + str(epoch) + '.pth')
                    validate(epoch)
                plt.figure()
                train_ax, = plt.plot(np.arange(0, epoch, 1), val_losses, label='Reg')
                val_ax, = plt.plot(np.arange(0, epoch, 1), train_losses, label='Elbo')
                # plt.ylim((0,1))
                plt.legend(handles=[train_ax, val_ax])
                plt.savefig(outputDir + "/Val/curves_epoch.png")

        elif opt.mode=="test":
            test()

