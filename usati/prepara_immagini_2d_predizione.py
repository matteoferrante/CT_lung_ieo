import pandas as pd
import numpy as np
import nrrd
import os
import glob
import cv2

#creo una cartella dove mettere le immagini
os.makedirs(r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\SLICES",exist_ok=True)

data=pd.read_csv(r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\Clinical_data_pseudoanonymized.csv",sep=";")


images_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"

roi_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"

patients=glob.glob(os.path.join(images_path,"*.nrrd"))
rois=glob.glob(os.path.join(roi_path,"*.nrrd"))

r,h=nrrd.read(patients[0])

for i in range(len(patients)):
    print(f"[INFO] patient {i}..",end=" - ")
    p,_=nrrd.read(patients[i])
    r,_=nrrd.read(rois[i])

    vals=(np.sum(np.sum(r,0),0)>0)
    print(f"found {np.sum(vals*1.)} slices of {p.shape[-1]}")
    for j in range(p.shape[-1]):
        if vals[j]:
            cv2.imwrite(os.path.join("D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\SLICES",f"{i}_{j}.png"),p[:,:,j])
            #mi chiedo se non convenga salvare le immagini come numpy array