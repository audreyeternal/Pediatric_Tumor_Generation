import numpy as np
import cv2
from glob import glob
from tqdm import tqdm 
from tqdm.contrib import tenumerate
import os
from hparam import hparams as hp

# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

pred_path = 'results/' + hp.Program_name +'/'
true_path = f"/host_project/SPADE/datasets/png_file/val_label_png_T1/"

pred_file = glob(pred_path+"*_int.png", recursive=True)
true_file = glob(true_path + "*", recursive=True)

dice_score_list = []
for i,(pred,true) in tenumerate(zip(sorted(pred_file),sorted(true_file))):
    y_pred = cv2.imread(pred,0)
    y_true = cv2.imread(true,0)
    y_true[y_true>3]=0
    y_true[y_true!=0] = 1 
    dice_score = dice(y_pred, y_true, k = 1)
    dice_score_list.append(dice_score)

final_dice_score = np.sum(np.array(dice_score_list))/len(pred_file)
print ("Dice Similarity: {}".format(final_dice_score))

