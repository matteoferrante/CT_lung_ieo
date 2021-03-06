# import the necessary packages
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self,dims,outputPath,dataKey="images",bufSize=1000):
        self.new=True
        #check to see if output path exist
        if os.path.exists(outputPath):
            print(F"[WARNING] File already exist. Append output. ")
            #raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)
            self.db=h5py.File(outputPath,"a")
            self.new=False
        else:
            self.db = h5py.File(outputPath, "w")

        ## POTREI LAVORARE QUA PER AGGIUNGERE OVERALL SURVIVAL E LINFONODI

        if self.new:
            #create 2 dataset one for images and one for dims
            self.data = self.db.create_dataset(dataKey, dims,
            dtype="int", maxshape=(None,dims[1],dims[2]))
            self.labels = self.db.create_dataset("labels", dims,
            dtype="int",maxshape=(None,dims[1],dims[2]))
            # store the buffer size, then initialize the buffer itself
            # along with the index into the datasets
            self.bufSize = bufSize
            self.buffer = {"data": [], "labels": []}
            self.idx = 0
        else:
            self.data=self.db[dataKey]
            self.labels=self.db["labels"]
            self.idx=len(self.db["images"])
            self.bufSize = bufSize
            self.buffer = {"data": [], "labels": []}

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()


    def flush(self):
        # write the buffers to disk then reset the buffer
        if self.new==False:
            self.data.resize(self.data.shape[0] +len(self.buffer["data"]), axis=0)
            self.labels.resize(self.labels.shape[0] + len(self.buffer["labels"]), axis=0)

        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",(len(classLabels),), dtype=dt)
        labelSet[:] = classLabels
    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()
        # close the dataset
        self.db.close()
