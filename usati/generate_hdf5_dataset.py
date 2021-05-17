"""Trasformo l'intero dataset in un file hdf5 slice per slice per rendere più veloce la lettura"""

import cv2
import h5py
import nrrd
import numpy as np
import glob
import os

import progressbar
from scipy.signal import argrelextrema

from utils.HDF5DatasetWriter import HDF5DatasetWriter


#sono tantissime immagini mi servirà un data generator


def load_image(path,std=False,targetDim=(112,112)):
    readdata, header = nrrd.read(path)
    return readdata

def load_annotation(path,targetDim=(112,112)):
    img=cv2.imread(path,0)
    img=cv2.resize(img,targetDim)
    img = np.expand_dims(img, -1)
    return img




#tentativo 1

dims=(256,256)

images_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"
roi_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"
output=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\hdf5_dataset"

images=glob.glob(os.path.join(images_path,"*.nrrd"))
masks=glob.glob(os.path.join(roi_path,"*.nrrd"))

X=[]
y=[]

seek_for_lungs=True

widgets = ["Loading nrrd image: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(images),widgets=widgets).start()


#se voglio fare resize lo metto qua
counter=0
for (c,(i,m)) in enumerate(list(zip(images,masks))):
    r,h=nrrd.read(i)
    r = cv2.resize(r.astype("float32"), dsize=dims, interpolation=cv2.INTER_CUBIC)
    r=np.swapaxes(r,0,-1)
    if seek_for_lungs:
        idx = argrelextrema(np.mean(r, (1, 2)), np.greater)
        min = idx[0][0]
        max = idx[0][-1]
        r = r[min:max]

    X.append(r)



    r, h = nrrd.read(m)
    r = cv2.resize(r.astype("float32"), dsize=dims, interpolation=cv2.INTER_CUBIC)
    r = np.swapaxes(r, 0, -1)
    if seek_for_lungs:
        # seleziono solo gli indici corretti
        r = r[min:max]
    y.append(r)
    pbar.update(c)



## QUA HDF5DatasetWriter
print(f"[INFO] adding images to dataset..")

#resize


#concatenation
X=np.concatenate(X,0)
y=np.concatenate(y,0)



#dataset = HDF5DatasetWriter((len(images)), dims=dims,output, dataKey="features", bufSize=args["buffer_size"]))
dataset=HDF5DatasetWriter((len(X),dims[0],dims[1]),outputPath=output,dataKey="images")


###IMPORTANTE QUA STO CARICANDO IMMAGINI E ROI MA NON LE INFORMAZIONI DI CLASSIFICAZIONE

dataset.add(X,y)
dataset.close()


## LOADING INFO

db = h5py.File(output)
print(db["images"][0].shape)

print(db["labels"].shape)