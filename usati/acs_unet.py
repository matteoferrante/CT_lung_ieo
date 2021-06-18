"""Script per provare l'implementazione ACS di pytorch
rif: https://github.com/M3DV/ACSConv
rif:https://arxiv.org/abs/1911.10477
"""
import time

from numpy.ma.bench import timer
from sklearn.model_selection import train_test_split

from acsconv.converters import ACSConverter
from acsconv.models.acsunet import ACSUNet
import torch
from torchvision.models import resnet18
from torchsummary import summary
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import numpy as np
import nrrd
import cv2
import glob


def main():
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    INIT_LR=1e-3
    BS=1
    EPOCHS=100
    DIMS=(256,256,192,1)


    model = smp.Unet(
        'efficientnet-b0',
        encoder_weights="imagenet",# choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        encoder_depth=3,
        decoder_channels=[256,128,64],
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    )

    #convert to ACS
    model_3d = ACSConverter(model)
    summary(model_3d, (DIMS[-1],DIMS[0],DIMS[1],DIMS[2]))

    ### path
    CT_paths=glob.glob(r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\CT\CT\*.nrrd")
    ROI_paths=glob.glob(r"D:\FISICA MEDICA\CT_LUNG\ieo_CT_lung_nrrd\ROI\ROI\*.nrrd")

    #unet_3d = ACSUnet(num_classes=2)

    ## datagenerator for pytorch
    class PytorchDataGenerator(torch.utils.data.Dataset):
        'Characterizes a dataset for PyTorch'

        def __init__(self, list_CTs, list_ROI, dim=(512, 512, 192, 1), upper=192,
                     shuffle=True,normalize=False):
            self.dim = dim
            self.list_ROI = list_ROI
            self.list_CTs = list_CTs
            self.shuffle = shuffle
            self.upper = upper
            self.type="train"
            self.normalize=normalize
        def __len__(self):
            # Denotes the total lenghtù
            return len(self.list_CTs)


        def __getitem__(self, index):
            # Generate one sample of data
            # Generate indexes of the batch
            #indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            #list_IDs_temp = [self.list_CTs[k] for k in indexes]
            #list_ROIs_temp = [self.list_ROI[k] for k in indexes]
            list_IDs_temp=self.list_CTs[index]
            list_ROIs_temp=self.list_ROI[index]
            # Generate data
            X, y = self.data_generation(list_IDs_temp, list_ROIs_temp)

    #roba sua
            #X = X.transpose((2, 0, 1))  # make image C x H x W
            X = torch.from_numpy(X)
            y=torch.from_numpy(y)
            # normalise image only if using mobilenet
            return X, y
        def data_generation(self,list_IDs_temp,list_ROIs_temp):
            #X = np.empty((self.batch_size, *self.dim))
            #y = np.empty((self.batch_size, *self.dim))

            # y = np.empty((self.batch_size), dtype=int)

            # Generate data

            n_x, _ = nrrd.read(list_IDs_temp) #è solo un elemento

            X = self.preprocess(n_x)
            n_y, _ = nrrd.read(list_ROIs_temp) #è solo uno
            # Store class
            y = self.preprocess(n_y)
            # for i, ID in enumerate(list_IDs_temp):
            #     # Store sample
            #
            #     n_x, _ = nrrd.read(ID)
            #
            #     X[i,] = self.preprocess(n_x)
            #
            #     ROI_ID = list_ROIs_temp[i]
            #     n_y, _ = nrrd.read(ROI_ID)
            #     # Store class
            #     y[i,] = self.preprocess(n_y)
            if self.normalize:
                X=(X-np.min(X))/(np.max(X)-np.min(X))
                y = (y - np.min(y)) / (np.max(y) - np.min(y))

            return X, y

        def preprocess(self, raw):

            # print(f"[DEBUG] {raw.shape}")
            # resizing
            if self.dim[:2] != raw.shape[:-1]:
                output = np.zeros((self.dim[0], self.dim[1], self.dim[2]))
                for i in range(raw.shape[-1]):
                    output[:, :, i] = cv2.resize(raw[:, :, i].astype('float32'), (self.dim[0], self.dim[1]))
                raw = output
            # print(f"[DEBUG - RES] {raw.shape}")
            # check the third dimension
            if raw.shape[2] < self.upper:

                ##pad the image with zeros

                if (self.upper - raw.shape[2]) % 2 == 0:
                    w = int((self.upper - raw.shape[-1]) / 2)
                    u = np.zeros((self.dim[0], self.dim[1], w))
                    raw = np.concatenate((u, raw, u), axis=-1)
                else:
                    w = int((self.upper - raw.shape[-1] - 1) / 2)
                    u = np.zeros((self.dim[0], self.dim[1], w))
                    d = np.zeros((self.dim[0], self.dim[1], w + 1))
                    raw = np.concatenate((u, raw, d), axis=-1)

            elif raw.shape[2] > self.upper:
                # crop the image along the third dimension
                dh = int(raw.shape[2] - self.upper / 2)
                raw = raw[:, :, dh:-dh]
            # print(f"[DEBUG2] {raw.shape}")

            return np.expand_dims(raw, 0)



    X_train, X_test, y_train, y_test = train_test_split(CT_paths, ROI_paths, test_size=0.30, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=24)


    train_set=PytorchDataGenerator(X_train,y_train,dim=DIMS)
    train_gen = torch.utils.data.DataLoader(train_set, batch_size=BS, shuffle=True)

    val_set=PytorchDataGenerator(X_val,y_val,dim=DIMS)
    val_gen = torch.utils.data.DataLoader(val_set, batch_size=BS, shuffle=True)

    test_set=PytorchDataGenerator(X_test,y_test,dim=DIMS)
    test_gen = torch.utils.data.DataLoader(test_set, batch_size=BS, shuffle=True)



    # Setup tensorboard
    if torch.cuda.is_available():
      dev = "cuda:0"
      print(f"[INFO] using {torch.cuda.get_device_name(0)}")
      device = torch.device(dev)
    else:
      dev = "cpu"
      device = torch.device(dev)



    ##OPT

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    best_val_acc = 0 # for model check pointing
    # Epoch loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        # Reset metrics
        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0.0
        val_correct = 0.0

        # Training loop
        model.train()
        for inputs, targets in train_gen:

            # use GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.float()
            targets = targets.view(-1,1).float()

            # Training steps
            optimizer.zero_grad() # clear gradients
            output = model(inputs) # forward pass: predict outputs for each image
            loss = loss(output, targets) # calculate loss
            loss.backward() # backward pass: compute gradient of the loss wrt model parameters
            optimizer.step() # update parameters
            train_loss += loss.item() * inputs.size(0) # update training loss
            train_correct += ((output>0.5) == targets).float().sum() # update training accuracy

        # Validation loop
        model.eval()
        for inputs, targets in val_gen:
            # use GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs=inputs.float()
            targets = targets.view(-1,1).float()

            # Validation steps
            with torch.no_grad(): #not calculating gradients every step
                output = model(inputs) # forward pass: predict outputs for each image
                loss = loss(output, targets) # calculate loss
                val_loss += loss.item() * inputs.size(0)  # update validation loss
                val_correct += ((output>0.5) == targets).float().sum() # update validation accuracy

        # calculate average losses and accuracy
        train_loss = train_loss/len(train_gen.sampler)
        val_loss = val_loss/len(val_gen.sampler)
        train_acc = train_correct / len(train_gen.sampler)
        val_acc = val_correct / len(val_gen.sampler)
        end_time = time.time() # get time taken for epoch

        # Display metrics at the end of each epoch.
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {val_loss} \tTraining Accuracy: {train_acc} \tValidation Accuracy: {val_acc} \t Time taken: {end_time - start_time}')

        # Log metrics to tensorboard
        #file_writer.add_scalar('Loss/train', train_loss, epoch)
        #file_writer.add_scalar('Loss/validation', val_loss, epoch)
        #file_writer.add_scalar('Accuracy/train', train_acc, epoch)
        #file_writer.add_scalar('Accuracy/validation', val_acc, epoch)
        #file_writer.add_scalar('epoch_time', end_time - start_time, epoch)

        # checkpoint if improved
        if val_acc>best_val_acc:
            state_dict = model.state_dict()
            torch.save(state_dict, "pytorch_acsunet"+'.pt')
            best_val_acc = val_acc





if __name__ == '__main__':
    main()