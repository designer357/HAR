import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
filename = "S1-ADL1.dat"
A = pd.read_csv(os.path.join(os.getcwd(),filename),delim_whitespace=True, header = None)
#B = A.dropna(axis=1,how='all')
#C = A.interpolate(method='linear', axis=0).ffill().bfill()

for tab in range(len(A.columns)):
    if float(A[tab].isnull().sum())/A[tab].count() > 0.3:
        del A[tab]
print(A.columns)
rows, columns = A[0].count(),len(A.columns)
XX = A.ix[90:120,9].interpolate(method='spline', order=3, s=2,axis=0).ffill().bfill()
#C = A.interpolate(method='spline', order=3, s=0,axis=0).ffill().bfill()
#temp = A.ix[0:rows - 1, 1].interpolate(method='spline', s=0, order=3, axis=0).ffill().bfill()

#for tab in range(2,10):
    #C = A.ix[0:rows-1,tab].interpolate(method='spline', s=0, order=3,axis=0).ffill().bfill()
    #C = A.interpolate(method='linear').ffill().bfill()
    #C = A.interpolate(method='spline', order=3, axis=0).ffill().bfill()
    #x = [i for i in range(len(A.ix[0:50000,0]))]
    #print(A.ix[0:50000,1])
    #print(str(tab) + " is processing!")
    #temp = np.concatenate((temp,C),axis=1)
    #print(str(tab) + " is finished!")
x = [i for i in range(len(A.ix[90:120,0]))]
plt.plot(x,A.ix[90:120,9],'b*',label = 'raw')
plt.plot(x,XX,'r.', label = 'spline')
plt.legend()
plt.show()
print(A.ix[90:120,9])
#C = A.interpolate(method='linear',axis=0).ffill().bfill()
#min_max_scaler = preprocessing.StandardScaler()
#np_scaled = min_max_scaler.fit_transform(C)
#df_normalized = pd.DataFrame(np_scaled)
#print(df_normalized)
#print(df_normalized.max() - df_normalized.min())
#print(df_normalized.ix[:,1:len(df_normalized)])