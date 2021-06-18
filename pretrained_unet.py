from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import unet_mini

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import nrrd
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
from kerassurgeon.operations import delete_layer, insert_layer

from utils.HDF5DatasetGenerator import HDF5DatasetGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

INIT_LR=1e-4
BS=16
DO=0.3
EPOCHS=10
VISTA="axial"


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

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


def change_model(model,new_input_shape=(None,256,256,1)):
    #replace input shape of first layer
    print(f"[INFO] Replacing {model._layers[0].name}")
    model._layers[0]._batch_input_shape=new_input_shape
    #rebuilt model
    new_model=keras.models.model_from_json(model.to_json())

    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print(f"[INFO] Loaded layer {layer.name}")
        except:
            print(f"[INFO] Could not load weights for {layer.name}")
    return new_model
### CARICO I DATI DA HDF5






#devo dividere train e test set e lanciare il training..

print(f"[INFO] Building the model..")

params = {'dim': (256,256),
          'batch_size': BS,
          'n_channels': 1,
          'shuffle': True}



model = unet_mini(n_classes=2,  input_height=256, input_width=256)

#training_generator = DataGenerator(IDs,**params)
training_generator=HDF5DatasetGenerator(rf"C:\Users\matte\Dataset\CT Lung IEO\train_ct_lung_{VISTA}_dataset.hdf5",BS,flatten=True)
val_generator=HDF5DatasetGenerator(rf"C:\Users\matte\Dataset\CT Lung IEO\val_ct_lung_{VISTA}_dataset.hdf5",BS,flatten=True)
### LAVORARE QUA


print(f"[INFO] Act on model to force input with one dimension")

model=change_model(model,new_input_shape=(None,params["dim"][0],params["dim"][1],params["n_channels"]))





#model = delete_layer(model.layers[0])
# inserts before layer 0
#model = insert_layer(model.layers[0], Input((params["dim"][0],params["dim"][1],params["n_channels"])))

# mean_iou
mean_iou=MyMeanIOU(num_classes=2)
mean_iou_keras = tf.keras.metrics.MeanIoU(num_classes=2)

opt=Adam(learning_rate=INIT_LR)

#model=Model(model.inputs,model.layers[-3].output)
model.compile(loss='binary_crossentropy', metrics=[dice_coef,mean_iou,mean_iou_keras])

model.summary()

print(f"[INFO] Fitting the model..")
#model.fit(X,y,epochs=EPOCHS)


checkpoint = ModelCheckpoint(filepath=fr'models\unet_{VISTA}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=3)

model.fit_generator(generator=training_generator.generator(),steps_per_epoch=training_generator.numImages // BS,epochs=EPOCHS,workers=-1,callbacks=[checkpoint,es],validation_data=val_generator.generator(),validation_steps=val_generator.numImages // BS)

#model.fit_generator(training_generator.generator(),callbacks=[checkpoint],workers=-1,use_multiprocessing=True,epochs=EPOCHS)

model.save(os.path.join("models",f"unet_{VISTA}.h5"))
print(f"[INFO] model saved in {os.path.join('models',f'unet_{VISTA}.h5')}")

