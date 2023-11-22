import os,sys
from glob import glob
import numpy as np
import SimpleITK as sitk
import scipy.misc as misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
from skimage import feature, filters, morphology, segmentation

def generateSketch(img, sigma=1, low_threshold=40, high_threshold=80):
    edges_in=sitk.CannyEdgeDetection(sitk.GetImageFromArray(img.astype(float)),lowerThreshold=low_threshold,upperThreshold=high_threshold)#,variance=[sigma, sigma, sigma])
    magnitude=sitk.SobelEdgeDetection(sitk.GetImageFromArray(img.astype(float)))
    edges_in=sitk.GetArrayFromImage(edges_in)
    magnitude=sitk.GetArrayFromImage(magnitude)
    magnitude=(magnitude-np.min(magnitude))/(np.max(magnitude)-np.min(magnitude))
    return (255*edges_in*magnitude)

def get_percent_of_voxels(img,percent):
    sorted=np.sort(img[img>0])
    length=sorted.shape[0]
    return sorted[int(round(length*percent))]

def Ostu_segment(img):
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(img_T1)
    myshow(sitk.LabelOverlay(img_T1_255, seg), "Otsu Thresholding")
    otsu_filter.GetThreshold()

def segmentVentricles(img, ventr_thresh=0.2):
    thresh_ventr = get_percent_of_voxels(img,ventr_thresh)
    ventr = np.zeros_like(img)
    ventr[img < thresh_ventr] = 1
    ventr[img == 0] = 0 #2022-03-01 ventricle:大于0，小于threshold
    ventr = morph.binary_fill_holes(ventr)
    brain_tissue = np.zeros_like(img)
    brain_tissue[img > 0] = 1
    brain_mask_eroded = morph.binary_erosion(brain_tissue, iterations=15)
    ventr = ventr * brain_mask_eroded
    brain_tissue = brain_tissue - ventr  
    segm_img = np.maximum(ventr.astype('uint8'), brain_tissue.astype('uint8') * 2) #2022-03-01 没太看懂?可能出现0,127,254三个值
    segm_img *= 127
    return segm_img

def save3DImage(img, path, inf_img=None):
    img = sitk.GetImageFromArray(img)
    if inf_img is not None:
        img.SetDirection(inf_img.GetDirection())
        img.SetOrigin(inf_img.GetOrigin())
        img.SetSpacing(inf_img.GetSpacing())
    sitk.WriteImage(img,path)

def read3DImage(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def normalizeImage(img, minValue=0,maxValue=255, valueType='uint8'):
    img=(img-np.min(img))/(np.max(img)-np.min(img))
    img=img*(maxValue-minValue)+minValue
    return img.astype(valueType)

def crop_BB(img, offset=3, crop_coords=None):
    if crop_coords is None:
        bb_coords = np.where(img > 0)
        x_min = np.min(bb_coords[0])
        x_min = max(x_min-offset, 0)
        x_max = np.max(bb_coords[0])
        x_max = min(x_max+offset, img.shape[0])
        y_min = np.min(bb_coords[1])
        y_min = max(y_min-offset, 0)
        y_max = np.max(bb_coords[1])
        y_max = min(y_max+offset, img.shape[1])
        if (len(img.shape)==3):
            z_min = np.min(bb_coords[2])
            z_min = max(z_min - offset, 0)
            z_max = np.max(bb_coords[2])
            z_max = min(z_max + offset, img.shape[2])
            return img[x_min:x_max, y_min:y_max, z_min:z_max],(x_min,x_max,y_min,y_max,z_min,z_max)
        return img[x_min:x_max, y_min:y_max], (x_min,x_max,y_min,y_max)
    else:
        x_min = crop_coords[0]
        x_min = max(x_min, 0)
        x_max = crop_coords[1]
        x_max = min(x_max, img.shape[0])
        y_min = crop_coords[2]
        y_min = max(y_min, 0)
        y_max = crop_coords[3]
        y_max = min(y_max, img.shape[1])
        if (len(img.shape) == 3):
            z_min = crop_coords[4]
            z_min = max(z_min, 0)
            z_max = crop_coords[5]
            z_max = min(z_max , img.shape[2])
            return img[x_min:x_max, y_min:y_max, z_min:z_max], crop_coords
        return img[x_min:x_max, y_min:y_max], crop_coords

def showTrainingData(training_data_loader):
    # test loaded data
    loaded_data = next(iter(training_data_loader))
    image_def = loaded_data['image_def']
    image_und = loaded_data['image_und']
    mask = loaded_data['mask']
    tumor = loaded_data['tumor']

    def show_image(input, title):
        def show_slices(slices, cmap):
            """ Function to display row of image slices """
            fig, axes = plt.subplots(1, len(slices))
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap=cmap, origin="lower")

        print(f"Feature batch shape: {input.size()}")
        if len(input.size()) > 4:
            numpy_input = input.numpy().squeeze(axis=0)
        else:
            numpy_input = input.numpy()
        for img_channel in range(numpy_input.shape[0]):
            img_3d = numpy_input[img_channel]
            print(f'Channel {img_channel} img:')
            H, W, C = img_3d.shape
            slice_0 = img_3d[int(H / 2), :, :]
            slice_1 = img_3d[:, int(W / 2), :]
            slice_2 = img_3d[:, :, int(C / 2)]
            show_slices([slice_0, slice_1, slice_2], "gray")
            plt.suptitle(title)
            plt.show()

    show_image(image_def, 'Deformed')
    show_image(image_und, 'Undeformed')
    show_image(mask, 'BrainMask')
    show_image(tumor, 'Tumor')

def showTestData(test_data_loader):
    # test loaded data
    loaded_data = next(iter(test_data_loader))
    image_def = loaded_data['image_def']
    # generated_tumor = loaded_data['generated_tumor']
    tumor = loaded_data['tumor']

    def show_image(input, title):
        def show_slices(slices, cmap):
            """ Function to display row of image slices """
            fig, axes = plt.subplots(1, len(slices))
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap=cmap, origin="lower")

        print(f"Feature batch shape: {input.size()}")
        if len(input.size()) > 4:
            numpy_input = input.numpy().squeeze(axis=0)
        else:
            numpy_input = input.numpy()
        for img_channel in range(numpy_input.shape[0]):
            img_3d = numpy_input[img_channel]
            print(f'Channel {img_channel} img:')
            H, W, C = img_3d.shape
            slice_0 = img_3d[int(H / 2), :, :]
            slice_1 = img_3d[:, int(W / 2), :]
            slice_2 = img_3d[:, :, int(C / 2)]
            show_slices([slice_0, slice_1, slice_2], "gray")
            plt.suptitle(title)
            plt.show()

    show_image(image_def, 'Deformed')
    # show_image(generated_tumor, 'Generated_tumor')
    show_image(tumor, 'Tumor')

def show3SlicesFromNumpy(img_3d, title):
    def show_slices(slices, cmap):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap=cmap, origin="lower")

    H, W, C = img_3d.shape
    slice_0 = img_3d[int(H / 2), :, :]
    slice_1 = img_3d[:, int(W / 2), :]
    slice_2 = img_3d[:, :, int(C / 2)]
    show_slices([slice_0, slice_1, slice_2], "gray")
    plt.suptitle(title)
    plt.show()