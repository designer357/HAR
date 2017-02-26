import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
train_file_list = ["S1-ADL1.dat","S1-ADL2.dat"]
test_file_list = ["S3-ADL1.dat","S3-ADL2.dat"]

folder = "Data"
def votting_label(mylist):
    N = len(mylist[0])
    output = np.empty(N)
    for i in range(N):
        u, indices = np.unique(mylist[:,i], return_inverse=True)
        output[i] = u[np.argmax(np.bincount(indices))]
    return output


def reConstruction(window_size,data,label):
    newdata = []
    newlabel = []
    L = len(data)
    interval = 1
    index = 0
    newdata_count = 0
    initial_value = -999
    while index+window_size < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        sequence = []
        temp = []
        for i in range(window_size):
            sequence.append(data[index+i])
            temp.append(label[index+i])
        newlabel[newdata_count] = votting_label(np.array(temp))
        index += interval
        newdata[newdata_count]=sequence
        newdata_count += 1
    return np.array(newdata),np.array(newlabel)
def loaddata(folder,train_file_list,test_file_list):
    trainX = []
    trainY = []
    testX = []
    testY = []
    input_folder = os.path.join(os.getcwd(),folder)
    for each_file in train_file_list:
        df = pd.read_csv(os.path.join(input_folder,each_file),delim_whitespace=True, header = None)
        #rows, columns = df[0].count(),len(df.columns)
        for tab in range(len(df.columns)):
            if float(df[tab].isnull().sum())/df[tab].count() > 1.0:
                del df[tab]
        df2 = df.interpolate(method='spline', order=3, s=0,axis=0).ffill().bfill()

        index = list(df2.columns)
        x_index = index[1:len(index)-2]
        y_index = index[len(index)-2:len(index)]
        x_, y_ = df2.ix[0:df2[0].count(),x_index],df2.ix[0:df2[0].count(),y_index]
        print(np.array(x_).shape)

        trainX.append(list(x_))
        trainY.append(list(y_))
    trainX2 = np.array(trainX)

    for each_file in test_file_list:
        df = pd.read_csv(os.path.join(input_folder, each_file), delim_whitespace=True, header=None)
        # rows, columns = df[0].count(),len(df.columns)
        for tab in range(len(df.columns)):
            if float(df[tab].isnull().sum()) / df[tab].count() > 0.3:
                del df[tab]
        df2 = df.interpolate(method='spline', order=3, s=0, axis=0).ffill().bfill()
        index = list(df2.columns)
        x_index = index[1:len(index) - 2]
        y_index = index[len(index) - 2:len(index)]
        x_, y_ = df2.ix[0:df2[0].count(), x_index], df2.ix[0:df2[0].count(), y_index]

        testX.append(np.array(x_))
        testY.append(np.array(y_))

    print(np.array(trainX).shape)

    return np.array(trainX),np.array(trainY),np.array(testX),np.array(testY)
X_train, y_train,X_test,y_test = loaddata(folder,train_file_list,test_file_list)
print(X_train.shape)
X,y = reConstruction(30,X_train,y_train)
print(X.shape)
print(y.shape)



