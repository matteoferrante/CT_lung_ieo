"""this code will use ACS to turn a pretrained 2D model like resnet or vggnet into a 3D one and perform a classification"""

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
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
import copy
import torchio as tio
import segmentation_models_pytorch as smp
import numpy as np
import nrrd
import cv2
import glob
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
import progressbar
###PARAMETERS

INIT_LR = 1e-4
BS = 1
EPOCHS = 10
DIMS = (224, 224, 224)
n_channels=3
num_classes = 2
feature_extract = True
MODEL = "vgg"


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]



### path
CT_paths = glob.glob(r"C:\Users\matte\Dataset\CT Lung IEO\CUBE\*.nrrd")
data = pd.read_csv(r"C:\Users\matte\Dataset\CT Lung IEO\Clinical_data_pseudoanonymized.csv",sep=";")


y=[]
target_variable="pN"
#get the labels
for i in CT_paths:
    idx=int(i[-8:-5])
    y.append(data[data["pz"]==idx][target_variable].values)

#Choose just pN for moment
lb=LabelBinarizer()
#labels=data["pN"]

# for i in labels:
#     if i <=np.percentile(labels,33):
#         y.append(0)
#     elif i>np.percentile(labels,33) and i<=np.percentile(labels,66):
#         y.append(1)
#     else:
#         y.append(2)

labels=np.array(y)
labels=lb.fit_transform(labels)
labels=to_categorical(labels,2)



####FUNCTIONS

# Setup tensorboard
if torch.cuda.is_available():
    dev = "cuda:0"
    print(f"[INFO] using {torch.cuda.get_device_name(0)}")
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)







def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f"[PHASE] {phase}")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            widgets = [
                ' [', progressbar.Timer(), '] ',
                progressbar.Bar(),
                ' (', progressbar.ETA(), ') ',
                ' (', progressbar.Variable("loss"), ') ',
                ' (', progressbar.Variable("acc"), ') '
            ]
            pbar = progressbar.ProgressBar(maxval=len(dataloaders[phase]),
                                           widgets=widgets).start()
            # Iterate over data.
            i=0
            for inputs, labels in dataloaders[phase]:
                i+=1


                inputs = inputs.to(device)
                labels = labels.to(device)

                #cast input to float
                inputs = inputs.float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.long())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.update(i,loss=running_loss,acc=running_corrects)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class PytorchDataGenerator(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_CTs, list_ROI, dim=(512, 512, 192, 1), upper=192,
                 shuffle=True, normalize=False,n_channels=1):
        self.dim = dim
        self.list_ROI = list_ROI
        self.list_CTs = list_CTs
        self.shuffle = shuffle
        self.upper = upper
        self.type = "train"
        self.normalize = normalize
        self.n_channels=n_channels

    def __len__(self):
        # Denotes the total lenghtù
        return len(self.list_CTs)

    def __getitem__(self, index):
        # Generate one sample of data
        # Generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_CTs[k] for k in indexes]
        # list_ROIs_temp = [self.list_ROI[k] for k in indexes]
        list_IDs_temp = self.list_CTs[index]
        list_ROIs_temp = self.list_ROI[index]
        # Generate data
        X, y = self.data_generation(list_IDs_temp, list_ROIs_temp)


        # roba sua
        # X = X.transpose((2, 0, 1))  # make image C x H x W
        X = torch.from_numpy(X)
        y = torch.from_numpy(np.asarray(y))
        # normalise image only if using mobilenet
        return X, y

    def data_generation(self, list_IDs_temp, list_ROIs_temp):
        # X = np.empty((self.batch_size, *self.dim))
        # y = np.empty((self.batch_size, *self.dim))

        # y = np.empty((self.batch_size), dtype=int)

        # Generate data

        n_x, _ = nrrd.read(list_IDs_temp)  # è solo un elemento

        X = self.preprocess(n_x)
        y=list_ROIs_temp
        #n_y, _ = nrrd.read(list_ROIs_temp)  # è solo uno
        # Store class
        #y = self.preprocess(n_y)
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
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
            #y = (y - np.min(y)) / (np.max(y) - np.min(y))

        if n_channels>1:
            X=np.repeat(X,n_channels,0)

        return X, y

    def preprocess(self, raw):

        resample = tio.Resample(raw.shape[0] / self.dim[0],image_interpolation='bspline')
        resampled = resample(np.expand_dims(raw,0))


        # #fix if image is bigger than fixed third dimension
        # if raw.shape[2] > self.upper:
        #     # crop the image along the third dimension
        #     dh = int(raw.shape[2] - self.upper / 2)
        #     raw = raw[:, :, dh:-dh]
        #
        # # resizing
        # if self.dim[:2] != raw.shape[:-1]:
        #     output = np.zeros((self.dim[0], self.dim[1], self.dim[2]))
        #     for i in range(raw.shape[-1]):
        #         try:
        #             output[:, :, i] = cv2.resize(raw[:, :, i].astype('float32'), (self.dim[0], self.dim[1]))
        #         except Exception as e:
        #             pass
        #             #print(e)
        #     raw = output
        #
        # # check the third dimension
        # if raw.shape[2] < self.upper:
        #
        #     ##pad the image with zeros
        #
        #     if (self.upper - raw.shape[2]) % 2 == 0:
        #         w = int((self.upper - raw.shape[-1]) / 2)
        #         u = np.zeros((self.dim[0], self.dim[1], w))
        #         raw = np.concatenate((u, raw, u), axis=-1)
        #     else:
        #         w = int((self.upper - raw.shape[-1] - 1) / 2)
        #         u = np.zeros((self.dim[0], self.dim[1], w))
        #         d = np.zeros((self.dim[0], self.dim[1], w + 1))
        #         raw = np.concatenate((u, raw, d), axis=-1)






        # print(f"[DEBUG2] {raw.shape}")

        return resampled


def main():
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()




    ##HERE START

    # Initialize the model for this run
    model, input_size = initialize_model(MODEL, num_classes, feature_extract, use_pretrained=True)

    #convert to ACS
    model_3d = ACSConverter(model)



    #unet_3d = ACSUnet(num_classes=2)

    ## datagenerator for pytorch


    X_train, X_test, y_train, y_test = train_test_split(CT_paths, labels, test_size=0.30, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=24)


    train_set=PytorchDataGenerator(X_train,y_train,dim=DIMS)
    train_gen = torch.utils.data.DataLoader(train_set, batch_size=BS, shuffle=True)

    val_set=PytorchDataGenerator(X_val,y_val,dim=DIMS)
    val_gen = torch.utils.data.DataLoader(val_set, batch_size=BS, shuffle=True)

    test_set=PytorchDataGenerator(X_test,y_test,dim=DIMS)
    test_gen = torch.utils.data.DataLoader(test_set, batch_size=BS, shuffle=True)

    dataloaders_dict={"train":train_gen,"val":val_gen}
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.



    ##FREEZE THE PARAMETERS


    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if "fc" in name:
                param.requires_grad=True
            else:
                param.requires_grad = False

            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)



#    summary(model_3d, (n_channels,DIMS[0],DIMS[1],DIMS[2]))

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam([
        dict(params=params_to_update, lr=INIT_LR),
    ])


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    #model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=EPOCHS, is_inception=(MODEL=="inception"))

    new_model = FeatureExtractor(model_3d)
    new_model.to(device)
    t_features,v_features,y_train,y_val=extract(new_model,dataloaders=dataloaders_dict)

    print("check")
    print(f"[INFO] {len(t_features)} shape: {t_features[0].shape} {len(y_train)} {y_train[0].shape}")

    t_features_df=pd.DataFrame(t_features)
    v_features_df=pd.DataFrame(v_features)

    t_features_df["target"]=y_train
    v_features_df["target"]=y_val

    t_features_df.to_csv("train_features_acs.csv",index=False)
    v_features_df.to_csv("val_features_acs.csv",index=False)

    #torch.save(model, 'models/acs_prediction.pk')


def extract(model,dataloaders):
    since = time.time()

    train_features=[]
    val_features=[]

    y_train=[]
    y_val=[]
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:

        widgets = [
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') '
        ]
        pbar = progressbar.ProgressBar(maxval=len(dataloaders[phase]),
                                       widgets=widgets).start()
        # Iterate over data.
        i=0
        for inputs, labels in dataloaders[phase]:
            i+=1

            pbar.update(i)
            inputs = inputs.to(device)
            #labels = labels.to(device)

            #cast input to float
            inputs = inputs.float()
            # zero the parameter gradients
            with torch.no_grad():
                # Extract the feature from the image
                feature = model(inputs)
                # Convert to NumPy Array, Reshape it, and save it to features variable
            if phase=="train":

                train_features.append(feature.cpu().detach().numpy().reshape(-1))
                y_train.append(labels)
            elif phase=="val":
                val_features.append(feature.cpu().detach().numpy().reshape(-1))
                y_val.append(labels)
    return train_features,val_features,y_train,y_val


if __name__ == '__main__':
    main()