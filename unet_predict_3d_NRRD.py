"""Questo codice importa e predice le ROI per immagini nrrd o dicom"""

import glob
import os
import pandas as pd
import cv2
import keras
import numpy as np
from imutils import paths
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import nrrd
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
from keras.models import load_model
from net.FullyConvNet import FullyConvNet

from scipy.signal import argrelextrema


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

import pydicom as dcm


base_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"
output_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\PRED"

MODEL=load_model(r"C:\Users\matte\PycharmProjects\CT_lung_ieo\models\my_unet_model.epoch04-loss0.00.hdf5",compile=False)
patients=os.listdir(base_path)
print(f"[INFO] Found {len(patients)} to analyze into {base_path}..")


def load_and_prepare_slices(paths,target_dim=(256,256),seek_for_lungs=True,dicom=False):

    ###LAVORARE QUA -> LO FACCIO CON GLI NRRD E NON CON I DICOM!

    X=[]
    #loading data
    if dicom:
        header=None #scegliere se mettere informazioni
        for i in paths:
            dcm_slice = dcm.dcmread(i)
            img = dcm_slice.pixel_array
            m = dcm_slice.RescaleSlope
            q = dcm_slice.RescaleIntercept
            img = m * img + q
            X.append(cv2.resize(img.astype("float32"),target_dim,interpolation=cv2.INTER_CUBIC))


        #concatenate data
        X = np.array(X)
    else:
        #nrrd read
        r,header=nrrd.read(paths)
        r=np.swapaxes(r,0,-1)


        for i in range(r.shape[0]):
            X.append(cv2.resize(r[i,:,:].astype("float32"), target_dim, interpolation=cv2.INTER_CUBIC))
        X = np.array(X)
    #resize e seek for lungs

    min=0
    max=len(X)

    if seek_for_lungs:
        idx = argrelextrema(np.mean(X, (1, 2)), np.greater)
        min = idx[0][0]
        max = idx[0][-1]
        X = X

    return X,min,max,header

def predict(x_p,min,max,targetDim=(256,256)):

    #prediction
    p=MODEL.predict(np.expand_dims(x_p[min:max],-1))

    #padding with 2D zeros
    top=np.zeros((min,targetDim[0],targetDim[1],1))

    b_pad=int(x_p.shape[0]-max)
    bottom=np.zeros((b_pad,targetDim[0],targetDim[1],1))

    return np.concatenate([top,p,bottom],axis=0)



def clean_roi(roi,min,max):

    cleaned_roi=np.zeros(roi.shape)

    for i in range(min,max):
        pred=roi[i,:,:,0]
        pred = (pred >= np.max(pred) - np.std(pred)) * 1.
        cleaned_roi[i,:,:,0]=pred

    # print(f"[DEBUG]")
    # for i in range(cleaned_roi.shape[0]):
    #     print(f"{np.max(cleaned_roi[i])}")
    return cleaned_roi

def enlarge(roi,findim=(512,512)):
    big_roi=np.zeros((roi.shape[0],findim[0],findim[1]))

    for i in range(roi.shape[0]):
        big_roi[i]=cv2.resize(roi[i],findim,cv2.INTER_NEAREST)
    big_roi=(big_roi>0.)*1.
    return big_roi

X_test=[]
lungs_extrema=[]  #store list on (min,max) values for each patient to reconstruct with correct padding
pred=[]

X_pred=[]

for p in patients:
    #patient_dir = os.path.join(base_path, p)

    #if os.path.isdir(patient_dir):

    print(f"[INFO] working on {p}")
    #enter into the directory


    #create the predicted roi directory
    #os.makedirs(os.path.join(patient_dir,"Pred"),exist_ok=True)

    #get the images:
    print(f"[INFO] loading acquisition of {p}")

    #slices=glob.glob(os.path.join(patient_dir,"Image","*.DCM"))
    patient=os.path.join(base_path,p)
    x_p,min_idx,max_idx,header=load_and_prepare_slices(patient)
    X_test.append(x_p)

    print(f"[INFO] prediction of ROI on {p}")
    #prediction
    roi_pred=predict(x_p,min_idx,max_idx)


    ##clean the roi


    roi_pred=clean_roi(roi_pred,min_idx,max_idx)
    X_pred.append(roi_pred)

    ##LAVORARE QUA -> DEVO INGRANDIRE LA ROI FINO ALLE DIMENSIONI ORIGINALI
    roi_pred=enlarge(roi_pred)

    #create nrrd and save prediction

    ## GIUSTO PER TENTATIVO
    roi_pred=np.swapaxes(roi_pred,0,-1)

    filename = os.path.join(output_path,f"{p}_predicted_roi.nrrd")
    nrrd.write(filename, roi_pred,header=header)



    #else:
    #    print(f"[INFO] Skipping {p} because is not a directory.")