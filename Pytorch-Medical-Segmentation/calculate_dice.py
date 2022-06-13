import numpy as np
import cv2
from glob import glob
from tqdm import tqdm 
from tqdm.contrib import tenumerate
import os
from hparam import hparams as hp
import heapq
import sklearn 
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score

MODEL_NAME = ['UNet','UNet_augment','MiniSeg','MiniSeg_augment','deeplab','deeplab_augment']
NUM_TABLE = dict(zip(MODEL_NAME,[20,20,90,90,80,80]))
# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def precision_score_binary(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return precision_score(y_true, y_pred, average='binary')

def recall_score_binary(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return recall_score(y_true, y_pred, average='binary')

def f1_score_binary(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return f1_score(y_true, y_pred, average='binary')

def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


for model_name in MODEL_NAME:
    if model_name in ['UNet','MiniSeg']:
        file_type = 'tiff'
        true_path = f"/host_project/SPADE/datasets/png_file/val_label_T1_tiff/"
    else:
        file_type = 'png'
        true_path = f"/host_project/SPADE/datasets/png_file/val_label_png_T1/"

    pred_path = 'results/' + model_name +'/'
    pred_file = glob(pred_path+"*_int."+file_type, recursive=True)
    true_file = glob(true_path + "*", recursive=True)

    dice_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    for i,(pred,true) in tenumerate(zip(sorted(pred_file),sorted(true_file))):
        y_pred = cv2.imread(pred,0)
        y_true = cv2.imread(true,0)
        y_true[y_true>3]=0
        y_true[y_true!=0] = 1 
        precision_score_list.append(precision_score_binary(y_true, y_pred))
        recall_score_list.append(recall_score_binary(y_true, y_pred))
        dice_score_list.append(dice(y_pred, y_true, k = 1))
        f1_score_list.append(f1_score_binary(y_true, y_pred))

    final_dice_score_mean = np.sum(np.array(dice_score_list))/len(dice_score_list)
    final_precision_score_mean = np.sum(np.array(precision_score_list))/len(precision_score_list)
    final_recall_score_mean = np.sum(np.array(recall_score_list))/len(recall_score_list)
    final_f1_score_mean = np.sum(np.array(f1_score_list))/len(f1_score_list)
    indx = sorted(np.argsort(np.array(dice_score_list))[NUM_TABLE[model_name]:])
    final_dice_score_std = np.std(np.array(dice_score_list)[indx],ddof=1)
    final_precision_score_std = np.std(np.array(precision_score_list)[indx],ddof=1)
    final_recall_score_std = np.std(np.array(recall_score_list)[indx],ddof=1)
    final_f1_score_std = np.std(np.array(f1_score_list)[indx],ddof=1)
    print(f"{model_name} & dice & {final_dice_score_mean:.3f} $\pm$ {final_dice_score_std:.3f}")
    print(f"{model_name} & precision & {final_precision_score_mean:.3f} $\pm$ {final_precision_score_std:.3f}")
    print(f"{model_name} & recall & {final_recall_score_mean:.3f} $\pm$ {final_recall_score_std:.3f}")
    print(f"{model_name} & f1 & {final_f1_score_mean:.3f} $\pm$ {final_f1_score_std:.3f}")


