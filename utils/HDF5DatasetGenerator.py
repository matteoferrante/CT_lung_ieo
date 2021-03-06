# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py
from keras_preprocessing.image import ImageDataGenerator


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, flatten=False,
                 aug=False, binarize=False, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
        self.flatten = flatten

        if self.aug:
            # generator for both images and masks
            data_gen_args = dict( rotation_range=45,
                                 width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True,
                                 vertical_flip=True)
            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)
            self.seed = 1

            self.augmentator = (image_datagen, mask_datagen)

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0
        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []
                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        # update the list of processed images
                        procImages.append(image)
                    # update the images array to be the processed
                    # images
                    images = np.array(procImages)
                    # if the data augmenator exists, apply it
                if self.aug:
                    image_datagen, mask_datagen = self.augmentator
                    (images, labels) = next(zip(image_datagen.flow(np.expand_dims(images, -1), seed=self.seed),
                                                mask_datagen.flow(np.expand_dims(labels, -1), seed=self.seed)))
                    images = np.squeeze(images, -1)
                    labels = np.squeeze(labels, -1)

                if self.flatten:
                    # images=np.resize(self.batchSize,images.shape[0]*images.shape[1])
                    labels = np.resize(self.batchSize, labels.shape[1] * labels.shape[2])
                # yield a tuple of images and labels

                images = np.expand_dims(images, -1)
                labels = np.expand_dims(labels, -1)

                yield (images, labels)
            epochs += 1

    def close(self):
        self.db.close()
#
#
# class HDF5DatasetGenerator:
#     def __init__(self, dbPath, batchSize, preprocessors=None,flatten=False,
#     aug=None, binarize=False, classes=2):
#         # store the batch size, preprocessors, and data augmentor,
#         # whether or not the labels should be binarized, along with
#         # the total number of classes
#         self.batchSize = batchSize
#         self.preprocessors = preprocessors
#         self.aug = aug
#         self.binarize = binarize
#         self.classes = classes
#         # open the HDF5 database for reading and determine the total
#         # number of entries in the database
#         self.db = h5py.File(dbPath)
#         self.numImages = self.db["labels"].shape[0]
#         self.flatten=flatten
#
#     def generator(self, passes=np.inf):
#         # initialize the epoch count
#         epochs = 0
#         # keep looping infinitely -- the model will stop once we have
#         # reach the desired number of epochs
#         while epochs < passes:
#             # loop over the HDF5 dataset
#             for i in np.arange(0, self.numImages, self.batchSize):
#                 # extract the images and labels from the HDF dataset
#                 images = self.db["images"][i: i + self.batchSize]
#                 labels = self.db["labels"][i: i + self.batchSize]
#                 # check to see if our preprocessors are not None
#                 if self.preprocessors is not None:
#                     # initialize the list of processed images
#                     procImages = []
#                     # loop over the images
#                     for image in images:
#                         # loop over the preprocessors and apply each
#                         # to the image
#                         for p in self.preprocessors:
#                             image = p.preprocess(image)
#                         # update the list of processed images
#                         procImages.append(image)
#                     # update the images array to be the processed
#                     # images
#                     images = np.array(procImages)
#                             # if the data augmenator exists, apply it
#                 if self.aug is not None:
#                     (images, labels) = next(self.aug.flow(images,labels, batch_size=self.batchSize))
#
#
#
#                 if self.flatten:
#                     #images=np.resize(self.batchSize,images.shape[0]*images.shape[1])
#                     labels=np.resize(self.batchSize,labels.shape[1]*labels.shape[2])
#                 # yield a tuple of images and labels
#
#
#
#                 images=np.expand_dims(images,-1)
#                 labels=np.expand_dims(labels,-1)
#
#                 yield (images, labels)
#             epochs+=1
#
#
#     def close(self):
#         self.db.close()
