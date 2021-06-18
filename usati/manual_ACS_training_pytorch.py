"""Ho già provato ACS ma non ho memoria, ho provato VNET ma uguale -> provo un approccio con 3 reti manuale"""
import h5py
import segmentation_models_pytorch as smp
import numpy as np
import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


# Setup tensorboard
print(f"[INFO] setup tensorboard")

if torch.cuda.is_available():
  dev = "cuda:0"
  print(f"[INFO] using {torch.cuda.get_device_name(0)}")
  device = torch.device(dev)
else:
  dev = "cpu"
  device = torch.device(dev)


INIT_LR=1e-3
BS=16
EPOCHS=40
DIMS=(256,256,1)
VISTA="axial"
ENCODER="mobilenet_v2"
ENCODER_WEIGHTS="imagenet"

train_dataset=rf"C:\Users\matte\Dataset\CT Lung IEO\train_ct_lung_{VISTA}_dataset.hdf5"
val_dataset=rf"C:\Users\matte\Dataset\CT Lung IEO\val_ct_lung_{VISTA}_dataset.hdf5"
test_dataset=rf"C:\Users\matte\Dataset\CT Lung IEO\test_ct_lung_{VISTA}_dataset.hdf5"


print(f"[INFO] path of \ntrain: {train_dataset}\nval: {val_dataset}\ntest: {test_dataset}")


model = smp.Unet(
    encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
    encoder_depth=4,
    decoder_channels=[256,128,16,8]
)

model.to(device)
model.double()
#summary(model,DIMS)



## datagenerator for pytorch
class PytorchDataGenerator(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dbPath, preprocessing=None,classes=1):
        self.classes = classes
        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
        self.preprocessing=preprocessing
    def __len__(self):
        # Denotes the total lenght
        return self.numImages


    def __getitem__(self, index):
        X = self.db["images"][index]
        y = self.db["labels"][index]
        X=np.expand_dims(X,0)
        y=np.expand_dims(y,0)

        if self.preprocessing is not None:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
            X=preprocessing_fn(X)
            y=preprocessing_fn(y)
        return X, y


print(f"[INFO] setting up data generators")


train_set=PytorchDataGenerator(train_dataset)
train_gen = torch.utils.data.DataLoader(train_set, batch_size=BS, shuffle=True)

val_set=PytorchDataGenerator(val_dataset)
val_gen = torch.utils.data.DataLoader(val_set, batch_size=BS, shuffle=True)

test_set=PytorchDataGenerator(test_dataset)
test_gen = torch.utils.data.DataLoader(test_set, batch_size=BS, shuffle=True)



print(f"[INFO] loss and optimizer definition")

loss = smp.utils.losses.DiceLoss()
metrics = [
  smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
  dict(params=model.parameters(), lr=INIT_LR),
])


#setting up train epochs and validation
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)






print(f"[INFO] Start training of model..")

max_score = 0

for i in range(0, EPOCHS):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_gen)
    valid_logs = valid_epoch.run(val_gen)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f'models/unet_pytorch_{VISTA}_best.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# load best saved checkpoint
print(f"[INFO] Resuming best model..")
best_model = torch.load(f'models/unet_pytorch_{VISTA}_best.pth')
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=device,
)

logs = test_epoch.run(test_gen)