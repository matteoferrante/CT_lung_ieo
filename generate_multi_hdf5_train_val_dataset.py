"""Come l'altro codice ma per fare coronale sogittale e assiale"""

"""Trasformo l'intero dataset in un file hdf5 slice per slice per rendere più veloce la lettura"""

import cv2
import h5py
import nrrd
import numpy as np
import glob
import os

import progressbar
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split

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


def generate_sagittal(X, y):
    #####################################################################
    print("[INFO] RUNNING SAGITTAL ANALYSIS")
    # SAGITTAL

    X = np.concatenate(X, -1)
    y = np.concatenate(y, -1)

    X = np.swapaxes(X, 0, -1)
    y = np.swapaxes(y, 0, -1)

    base_out = r"C:\Users\matte\Dataset\CT Lung IEO"
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

    # dataset = HDF5DatasetWriter((len(images)), dims=dims,output, dataKey="features", bufSize=args["buffer_size"]))

    train_dataset = HDF5DatasetWriter((len(X_train), dims[0], dims[1]),
                                      outputPath=os.path.join(base_out, "train_ct_lung_sagittal_dataset.hdf5"),
                                      dataKey="images")

    # test_dataset = HDF5DatasetWriter((len(X_test), dims[0], dims[1]),
    #                                  outputPath=os.path.join(base_out, "test_ct_lung_sagittal_dataset.hdf5"),
    #                                  dataKey="images")

    val_dataset = HDF5DatasetWriter((len(X_val), dims[0], dims[1]),
                                    outputPath=os.path.join(base_out, "val_ct_lung_sagittal_dataset.hdf5"),
                                    dataKey="images")

    ###IMPORTANTE QUA STO CARICANDO IMMAGINI E ROI MA NON LE INFORMAZIONI DI CLASSIFICAZIONE

    print(f"[INFO] Writing training dataset")
    train_dataset.add(X_train, y_train)

    train_dataset.close()
    # print(f"[INFO] Writing test dataset")
    #
    # test_dataset.add(X_test, y_test)
    # test_dataset.close()

    print(f"[INFO] Writing validation dataset")
    val_dataset.add(X_val, y_val)

    val_dataset.close()


#####################################################################


def generate_coronal(X, y):
    print(F"[INFO] RUNNING CORONAL ANALYSIS")

    X = np.concatenate(X, 1)
    y = np.concatenate(y, 1)

    X = np.swapaxes(X, 0, 1)
    y = np.swapaxes(y, 0, 1)

    base_out = r"C:\Users\matte\Dataset\CT Lung IEO"
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

    # dataset = HDF5DatasetWriter((len(images)), dims=dims,output, dataKey="features", bufSize=args["buffer_size"]))

    train_dataset = HDF5DatasetWriter((len(X_train), dims[0], dims[1]),
                                      outputPath=os.path.join(base_out, "train_ct_lung_coronal_dataset.hdf5"),
                                      dataKey="images")

    #test_dataset = HDF5DatasetWriter((len(X_test), dims[0], dims[1]),
 #                                    outputPath=os.path.join(base_out, "test_ct_lung_coronal_dataset.hdf5"),
#                                     dataKey="images")

    val_dataset = HDF5DatasetWriter((len(X_val), dims[0], dims[1]),
                                    outputPath=os.path.join(base_out, "val_ct_lung_coronal_dataset.hdf5"),
                                    dataKey="images")

    ###IMPORTANTE QUA STO CARICANDO IMMAGINI E ROI MA NON LE INFORMAZIONI DI CLASSIFICAZIONE

    print(f"[INFO] Writing training dataset")
    train_dataset.add(X_train, y_train)

    train_dataset.close()
    # print(f"[INFO] Writing test dataset")
    #
    # test_dataset.add(X_test, y_test)
    # test_dataset.close()

    print(f"[INFO] Writing validation dataset")
    val_dataset.add(X_val, y_val)

    val_dataset.close()



def generate_axial(X, y):
    print(F"[INFO] RUNNING AXIAL ANALYSIS")

    X = np.concatenate(X, 0)
    y = np.concatenate(y, 0)

    #X_c = np.swapaxes(X, 0, 1)
    #y_c = np.swapaxes(y, 0, 1)

    base_out = r"C:\Users\matte\Dataset\CT Lung IEO"
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

    # dataset = HDF5DatasetWriter((len(images)), dims=dims,output, dataKey="features", bufSize=args["buffer_size"]))

    train_dataset = HDF5DatasetWriter((len(X_train), dims[0], dims[1]),
                                      outputPath=os.path.join(base_out, "train_ct_lung_axial_dataset.hdf5"),
                                      dataKey="images")

    #test_dataset = HDF5DatasetWriter((len(X_test), dims[0], dims[1]),
 #                                    outputPath=os.path.join(base_out, "test_ct_lung_coronal_dataset.hdf5"),
#                                     dataKey="images")

    val_dataset = HDF5DatasetWriter((len(X_val), dims[0], dims[1]),
                                    outputPath=os.path.join(base_out, "val_ct_lung_axial_dataset.hdf5"),
                                    dataKey="images")

    ###IMPORTANTE QUA STO CARICANDO IMMAGINI E ROI MA NON LE INFORMAZIONI DI CLASSIFICAZIONE

    print(f"[INFO] Writing training dataset")
    train_dataset.add(X_train, y_train)

    train_dataset.close()
    # print(f"[INFO] Writing test dataset")
    #
    # test_dataset.add(X_test, y_test)
    # test_dataset.close()

    print(f"[INFO] Writing validation dataset")
    val_dataset.add(X_val, y_val)

    val_dataset.close()


### QUA INIZIA IL CODICE


#tentativo 1

dims=(256,256)

#uso dataset aggiuntivo
#images_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"
#roi_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"

images_path=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\IEO_OD\CT"
roi_path=r"D:\FISICA MEDICA\CT_LUNG\altri_dati\IEO_OD\ROI"


output=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\hdf5_dataset"

images=glob.glob(os.path.join(images_path,"*.nrrd"))
masks=glob.glob(os.path.join(roi_path,"*.nrrd"))
IDX=220 #reserve all idxs over 220 for testing

X=[]
y=[]

seek_for_lungs=True

widgets = ["Loading nrrd image: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(images),widgets=widgets).start()


#se voglio fare resize lo metto qua
counter=0
dmin=2
dmax=1
upper=256

def pad(raw,upper):


    if raw.shape[0] < upper:

        ##pad the image with zeros

        if (upper - raw.shape[0]) % 2 == 0:

            w = int((upper - raw.shape[0]) / 2)
            u = np.zeros((w,dims[0], dims[1]))
            raw = np.concatenate((u, raw, u), axis=0)
        else:
            w = int((upper - raw.shape[0] - 1) / 2)
            u = np.zeros((w,dims[0], dims[1]))
            d = np.zeros((w+1,dims[0], dims[1]))
            raw = np.concatenate((u, raw, d), axis=0)
    elif raw.shape[0] > upper:
        # crop the image along the third dimension
        dh = int(raw.shape[0] - upper / 2)
        raw = raw[dh:-dh, :, :]
    return raw

#questo codice è molto pesante, se dovesse dare errori di memoria caricare solo una frazione del dataset per volta [start:stop] e rilanciarlo più volte
for (c,(i,m)) in enumerate(list(zip(images,masks))[110:160]):
    r,h=nrrd.read(i)
    r = cv2.resize(r.astype("float32"), dsize=dims, interpolation=cv2.INTER_CUBIC)
    #recast to int
    r=r.astype("int8")
    r=np.swapaxes(r,0,-1)
    if seek_for_lungs:
        idx = argrelextrema(np.mean(r, (1, 2)), np.greater)
        min = idx[0][0]
        max = idx[0][-1]
        r = r[min+dmin:max-dmax]
    r=pad(r,upper)
    X.append(r)



    r, h = nrrd.read(m)
    r = cv2.resize(r.astype("float32"), dsize=dims, interpolation=cv2.INTER_CUBIC)
    r=r.astype("int8")
    r = np.swapaxes(r, 0, -1)
    if seek_for_lungs:
        # seleziono solo gli indici corretti
        r = r[min+dmin:max-dmax]
    r = pad(r, upper)
    y.append(r)
    pbar.update(c)



## QUA HDF5DatasetWriter
print(f"[INFO] adding images to dataset..")

generate_sagittal(X,y)

#resize

#concatenation
act_dim=(len(X),dims[0],dims[1])  #buono per assiale

#devo matchare le dimensioni




print(f"[INFO] Programs ended.")
# ## LOADING INFO
#
# db = h5py.File(output)
# print(db["images"][0].shape)
#
# print(db["labels"].shape)