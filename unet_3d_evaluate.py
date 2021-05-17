
import glob
import os
import pandas as pd
import cv2
import numpy as np
from imutils import paths

import matplotlib.pyplot as plt
import nrrd


def dice(pred, true, k = 1):


    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))

    ior=intersection/np.sum(true)
    return dice,ior/2

#list rois
roi_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"
rois=glob.glob(os.path.join(roi_path,"*.nrrd"))

pred_path="D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\PRED"
predictions=glob.glob(os.path.join(pred_path,"*.nrrd"))



dices=[]
iors=[]
for (p,r) in zip(rois,predictions):
    #load pred and roi
    pred,_=nrrd.read(p)
    roi,_=nrrd.read(r)
    d,ior=dice(pred,roi)
    print(f"[INFO] {p} dice: {d} - intersection_over_roi: {ior} ")
    dices.append(d)
    iors.append(ior)
df=pd.DataFrame()
df["Dice"]=dices
df["Intersection_over_roi"]=iors

df.to_csv("unet_evaluation.csv")