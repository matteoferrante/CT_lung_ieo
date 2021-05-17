import keras
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Cropping2D, ZeroPadding2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten, Concatenate
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate


def conv_module(x, n_filter, kX, kY, dr, chanDim=-1, padding="same", do=0.3):
    # define a (CONV => BN => RELU pattern => DROPOUT)*2
    x = Conv2D(n_filter, (kX, kY), dilation_rate=dr, padding=padding)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    x = Dropout(do)(x)

    return x


class FullyConvNet:

    @staticmethod
    def build_model(inputSize,name="fully_convnet"):
        i = Input(shape=inputSize)
        # conv_1
        c1 = conv_module(i, 32, 3, 3, (1, 1))

        c1_1 = conv_module(c1, 32, 3, 3, (1, 1))

        # conv2

        c2 = conv_module(c1_1, 32, 3, 3, (2, 2))
        c2_o = conv_module(c2, 32, 3, 3, (2, 2))

        # da c2_o inizia la disambiguazione

        # conv3
        c3 = conv_module(c2_o, 32, 3, 3, (1, 1))
        c3_o = conv_module(c3, 32, 1, 1, (1, 1))

        # conv4
        c4 = conv_module(c2_o, 32, 3, 3, (6, 6))
        c4_o = conv_module(c4, 32, 3, 3, (6, 6))

        # conv5
        c5 = conv_module(c2_o, 32, 3, 3, (12, 12))
        c5_o = conv_module(c5, 32, 3, 3, (12, 12))

        # conv6
        c6 = conv_module(c2_o, 32, 3, 3, (18, 18))
        c6_o = conv_module(c6, 32, 3, 3, (18, 18))

        # conv7
        c7 = conv_module(c2_o, 32, 3, 3, (24, 24))
        c7_o = conv_module(c7, 32, 3, 3, (24, 24))

        # concatenate 3-4-5-6-7

        conc = concatenate([c3_o, c4_o, c5_o, c6_o, c7_o])

        c8 = conv_module(conc, 64, 1, 1, (1, 1))


        #pred = Conv2D(2, (1, 1), padding="same")(c8)
        ##modifica
        pred = Conv2D(1, (1, 1), padding="same")(c8)  #modello binario

        model = Model(i, pred, name=name)

        return model
