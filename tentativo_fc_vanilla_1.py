"Provo con un approccio semplice: carico tutte le immagini considerando le slices come indipendenti, carico le roi e provo ad allenare una fully conv net"
import glob
import os

import cv2
import keras
import numpy as np
from imutils import paths
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from net.FullyConvNet import FullyConvNet
import nrrd

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

INIT_LR=1e-3
BS=4
DO=0.3
EPOCHS=10

images_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT"
roi_path=r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI"

#sono tantissime immagini mi servirà un data generator


def load_image(path,std=False,targetDim=(112,112)):
    readdata, header = nrrd.read(path)
    return readdata

def load_annotation(path,targetDim=(112,112)):
    img=cv2.imread(path,0)
    img=cv2.resize(img,targetDim)
    img = np.expand_dims(img, -1)
    return img



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


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)
        y= np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            #print(f"[DEBUG] reading {ID[0]} \t {ID[1]}")
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = load_image(ID[0])

            # Store class
            #y[i] = self.labels[ID[1]]
            y[i] = load_image(ID[1])

        return X, y



### elenco le immagini nelle cartelle

images=glob.glob(os.path.join(images_path,"*.nrrd"))
masks=glob.glob(os.path.join(roi_path,"*.nrrd"))


# a questo punto ho i percorsi delle immagini e le roi

#devo dividere train e test set e lanciare il training..
