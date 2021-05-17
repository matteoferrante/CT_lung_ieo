"Provo con un approccio semplice: carico tutte le immagini considerando le slices come indipendenti, carico le roi e provo ad allenare una fully conv net"

"COMMENTO: USO HDF5 e slice"


import glob
import os

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

from net.FullyConvNet import FullyConvNet
from utils.HDF5DatasetGenerator import HDF5DatasetGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

INIT_LR=1e-3
BS=4
DO=0.3
EPOCHS=10



def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

### CARICO I DATI DA HDF5






#devo dividere train e test set e lanciare il training..

print(f"[INFO] Building the model..")

params = {'dim': (256,256),
          'batch_size': BS,
          'n_channels': 1,
          'shuffle': True}

#training_generator = DataGenerator(IDs,**params)
training_generator=HDF5DatasetGenerator(r"C:\Users\matte\Dataset\CT Lung IEO\train_ct_lung_dataset.hdf5",BS)
val_generator=HDF5DatasetGenerator(r"C:\Users\matte\Dataset\CT Lung IEO\val_ct_lung_dataset.hdf5",BS)
### LAVORARE QUA

model=FullyConvNet.build_model((params["dim"][0],params["dim"][1],params["n_channels"]))

model.summary()

opt=Adam(learning_rate=INIT_LR)

model.compile(optimizer=opt,loss=dice_coef_loss)


print(f"[INFO] Fitting the model..")
#model.fit(X,y,epochs=EPOCHS)


checkpoint = ModelCheckpoint(filepath='models\my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

model.fit_generator(generator=training_generator.generator(),steps_per_epoch=training_generator.numImages // BS,epochs=EPOCHS,workers=-1,callbacks=[checkpoint],validation_data=val_generator.generator(),validation_steps=val_generator.numImages // BS)

#model.fit_generator(training_generator.generator(),callbacks=[checkpoint],workers=-1,use_multiprocessing=True,epochs=EPOCHS)

model.save(os.path.join("models","fullyconv_vanilla_2.h5"))
print(f"[INFO] model saved in {os.path.join('models','fullyconv_vanilla_2.h5')}")


