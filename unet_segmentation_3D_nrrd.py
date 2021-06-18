"""Questo codice importa e predice le ROI per immagini nrrd o dicom"""

import glob
import os
import pandas as pd
import cv2
import keras
import numpy as np
import progressbar
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
import cc3d
from scipy.signal import argrelextrema


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

import pydicom as dcm


DATASET=1 #1 il primo, 2 gli altri


if DATASET==1:

    #dataset 1
    ct_paths=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"
    roi_paths=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"
    mask_paths=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\MASK\MASK"
    target_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\PRED_ACS"

elif DATASET==2:
    #dataset 2
    ct_paths=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\IEO_OD\CT"
    roi_paths=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\IEO_OD\ROI"
    mask_paths=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\IEO_OD\BOX"
    target_path=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\CT\PRED_ACS"



MODEL_AXIAL=load_model(r"models/unet_axial.hdf5",compile=False)
MODEL_SAGITTAL=load_model(r"models/unet_sagittal.hdf5",compile=False)
MODEL_CORONAL=load_model(r"models/unet_coronal.hdf5",compile=False)


# a volte non riesce a farli tutti, ne faccio 90 per volta
START=220
END=START+3

use_mask=True

def load_and_prepare(ct_path, roi_path, mask_path, target_dims=(256, 256), third=256, do_pad=True, apply_mask=False):
    ct, hh = nrrd.read(ct_path)
    roi, hh = nrrd.read(roi_path)
    if apply_mask:
        mask, hh = nrrd.read(mask_path)
    else:
        mask=None

    # resize
    if ct.shape[1:] != target_dims:
        ct = cv2.resize(ct.astype("float32"), dsize=target_dims, interpolation=cv2.INTER_CUBIC)

    if roi.shape[1:] != target_dims:
        roi = cv2.resize(roi.astype("float32"), dsize=target_dims, interpolation=cv2.INTER_CUBIC)

    if apply_mask:
        if mask.shape[1:] != target_dims:
            mask = cv2.resize(mask.astype("float32"), dsize=target_dims, interpolation=cv2.INTER_CUBIC)

    # swap axes to have (#of image, dim_x,dim_y)
    ct = np.swapaxes(ct, 0, -1)
    roi = np.swapaxes(roi, 0, -1)
    if apply_mask:
        mask = np.swapaxes(mask, 0, -1)

        mask = (mask > 0.) * 1.

    info = None
    if do_pad:
        # padding images
        ct, info = pad(ct, target_dims, third)
        roi, info = pad(roi, target_dims, third)
        if apply_mask:
            mask, _ = pad(mask, target_dims, third)

    if apply_mask:
        ct = ct * mask

    return ct, roi, mask, info, hh


def pad(raw, dims=(256, 256), upper=256):
    cropped = False
    w = None
    dh = None
    updownsamepadding = None

    if raw.shape[0] < upper:

        ##pad the image with zeros

        if (upper - raw.shape[0]) % 2 == 0:

            w = int((upper - raw.shape[0]) / 2)
            u = np.zeros((w, dims[0], dims[1]))
            raw = np.concatenate((u, raw, u), axis=0)
            updownsamepadding = True  # same dim for padding up and down
        else:
            w = int((upper - raw.shape[0] - 1) / 2)
            u = np.zeros((w, dims[0], dims[1]))
            d = np.zeros((w + 1, dims[0], dims[1]))
            raw = np.concatenate((u, raw, d), axis=0)
            updownsamepadding = False  # padding down with a +1 dim
    elif raw.shape[0] > upper:
        # crop the image along the third dimension
        dh = int(raw.shape[0] - upper / 2)
        raw = raw[dh:-dh, :, :]
        cropped = True

    info = {"cropped": cropped, "w": w, "dh": dh, "updownsamepadding": updownsamepadding}

    return raw, info


def raw_predict(x_p):
    # axial prediction
    a = MODEL_AXIAL.predict(np.expand_dims(x_p, -1))

    # coronal
    x_c = np.swapaxes(x_p, 0, 1)
    c = MODEL_CORONAL.predict(np.expand_dims(x_c, -1))
    c = np.swapaxes(c, 0, 1)

    # sagittal

    x_s = np.swapaxes(x_p, 0, 2)
    s = MODEL_SAGITTAL.predict(np.expand_dims(x_s, -1))
    s = np.swapaxes(s, 0, 2)

    return a, c, s


def clean_roi(roi, info=None, mask=None, method="thr",thr=0.1,normalize=True):
    cleaned_roi = np.zeros(roi.shape)
    start = 0
    stop = roi.shape[0]

    if normalize:
        roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))

    if info is not None:
        start = info["w"]
        stop = roi.shape[0] - info["w"]

    if mask is not None:
        roi = np.expand_dims(roi[:, :, :, 0] * mask, -1)

    for i in range(start, stop):
        pred = roi[i, :, :, 0]
        if method=="percentile":
            pred = (pred >= np.percentile(pred, thr)) * 1.
        elif method=="thr":
            pred=(pred>thr)*1.
        cleaned_roi[i, :, :, 0] = pred

    #per sicurezza
    if mask is not None:
        cleaned_roi = np.expand_dims(cleaned_roi[:, :, :, 0] * mask, -1)

    return cleaned_roi


def enlarge(roi, findim=(512, 512)):
    big_roi = np.zeros((roi.shape[0], findim[0], findim[1]))

    for i in range(roi.shape[0]):
        big_roi[i] = cv2.resize(roi[i], findim)
    big_roi = (big_roi > 0.) * 1.
    return big_roi

def prepare_roi_for_presentation(roi, target_path, info, header,connected=None):




    roi_pred = np.squeeze(roi)

    if connected is not None:
        #search for connected componenents
        roi_pred=keep_connected_components(roi_pred,connected)

    # info
    if info["w"] is not None:
        w = info["w"]
        if info["updownsamepadding"]:
            roi_pred = roi_pred[w:-w]
        else:
            roi_pred = roi_pred[w + 1:-w]
    if info["cropped"]:
        u = np.zeros((info["dh"], roi_pred.shape[1], roi_pred.shape[2]))
        roi_pred = np.concatenate((u, roi_pred, u), axis=0)
    roi_pred = enlarge(roi_pred)
    roi_pred = np.swapaxes(roi_pred, 0, -1)
    #print(f"[INFO] Saving {target_path} with shape: {roi_pred.shape}")
    nrrd.write(target_path, roi_pred, header=header)




def keep_connected_components(roi,conn=6):

    roi=roi.astype("int")
    labels_out, N = cc3d.connected_components(roi, connectivity=conn, return_N=True)
    connected = []

    if N>0:
        for lab in range(N):
            s = np.sum((labels_out == lab) * 1.)
            connected.append(s)

        max_lbl = np.argmax(connected[1:]) + 1
        return (labels_out==max_lbl)*1.
    else:
        print(f"[WARNING] No connected components found")
        return roi*1.





X=[]
y=[]
masks=[]
headers=[]
infos=[]
ct_list=glob.glob(os.path.join(ct_paths,"*.nrrd"))
roi_list=glob.glob(os.path.join(roi_paths,"*.nrrd"))
if use_mask:
    mask_list=glob.glob(os.path.join(mask_paths,"*.nrrd"))
else:
    mask_list=[None]*len(ct_list)

for (i,c) in enumerate(ct_list[START:END]):
    idx=c[-8:-5]

    #sui dati normali usare ROI

    r=os.path.join(roi_paths,f"ROI{idx}.nrrd")
    #usare box o mask
    if use_mask:
        m=os.path.join(mask_paths,f"MASK{idx}.nrrd")
    else:
        m=None
    print(f"[INFO] Running  {i}/{len(ct_list)}")
    print(r,c,m)
    ct,roi,mask,info,header=load_and_prepare(c,r,m,apply_mask=use_mask)
    X.append(ct)
    y.append(roi)
    masks.append(mask)
    infos.append(info)
    headers.append(header)

print("[INFO] Prediction from axial, sagittal and coronal slices")
widgets = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') '
]
pbar = progressbar.ProgressBar(maxval=len(X),
                               widgets=widgets).start()


start_idx=ct_list[START][-8:-5]
print(start_idx)
start_idx=eval(start_idx)
for (i,x_i) in enumerate(X):

    a, c, s = raw_predict(x_i)

    if use_mask:
        mask_to_apply=masks[i]
    else:
        mask_to_apply=None
    aa = clean_roi(a, mask=mask_to_apply)
    cc = clean_roi(c, mask=mask_to_apply)
    ss = clean_roi(s, mask=mask_to_apply)

    acs_sum = aa + cc + ss
    #acs_prod = aa * cc * ss
    acs_sum_thr = (acs_sum > 0) * 1.

    target=os.path.join(target_path,f"PRED{i+start_idx:03}.nrrd")
    prepare_roi_for_presentation(acs_sum_thr, target, infos[i], header=headers[i],connected=26)
    pbar.update(i)

print(f"[END]")