

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
from keras.models import load_model
from net.FullyConvNet import FullyConvNet
from utils.HDF5DatasetGenerator import HDF5DatasetGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)


BS=4

### test generator
test_generator=HDF5DatasetGenerator(r"C:\Users\matte\Dataset\CT Lung IEO\test_ct_lung_dataset.hdf5",BS)


#load model
model=load_model(r"models/venuto_male_primo_tentativo.hdf5")

pred=model.predict_generator(test_generator.generator(),steps=test_generator.numImages/BS)