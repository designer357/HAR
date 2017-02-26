import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
filename = "S1-ADL1.dat"
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
def loaddata(folder,filename):
    input_folder = os.path.join(os.getcwd(),folder)
    df = pd.read_csv(os.path.join(input_folder,filename),delim_whitespace=True, header = None)
    #rows, columns = df[0].count(),len(df.columns)
    for tab in range(len(df.columns)):
        if float(df[tab].isnull().sum())/df[tab].count() > 0.3:
            del df[tab]
    df2 = df.interpolate(method='spline', order=3, s=0,axis=0).ffill().bfill()

    index = list(df2.columns)
    x_index = index[1:len(index)-2]
    y_index = index[len(index)-2:len(index)]
    x, y = df2.ix[0:df2[0].count(),x_index],df2.ix[0:df2[0].count(),y_index]
    x = np.array(x)
    y = np.array(y)
    return x,y
X, y = loaddata(folder,filename)
X,y = reConstruction(30,X,y)
print(X.shape)
print(y.shape)



