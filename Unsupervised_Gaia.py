import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
import os, contextlib, sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, quantile_transform
from datetime import datetime
from sklearn.cluster import KMeans

# Sets the directory to the current directory
os.chdir(sys.path[0])

random_state = 27

def read_in(file):
    return Table.read(filename, format='fits').to_pandas()

def class_extraction(table, nameline):
    table[nameline] = [str(n) for n in table[nameline]]
    table[nameline] = [n.replace('b','') for n in table[nameline]]
    table[nameline] = [n.replace(' ','') for n in table[nameline]]
    table[nameline] = [n.replace("'","") for n in table[nameline]]
    with_class = table[nameline] != ''
    table = table.loc[with_class]
    return table, with_class, np.unique(table[nameline])
    


# Reading the data and converting to a pandas data frame - works more smoothly with sklearn
filename = 'GaiaSDSSUKIDSSAllWISE.fits'

data_table = read_in(filename)

data_table_with_names, with_class, names = class_extraction(data_table, 'SpCl_s_gs_gsu')



# Creating a copy for the preprocessing

data_scale = data_table.copy()


# Preprocessing the data (not any data of string-type) via the quantile transformer

data_scale = data_scale.dropna(axis=1)

non_string = data_scale.dtypes != object

data_scale = data_scale.loc[:, non_string]

#scaler = StandardScaler(copy=True)
#data_scale.loc[:, non_string] = scaler.fit_transform(data_scale.loc[:, non_string])

scaler = 'Quantile'

data_scale = quantile_transform(data_scale, copy=True)


#print('TSNE-ing')
# Clustering the data via t-SNE

#data_embedded = TSNE(n_components=2,n_jobs=-1,).fit_transform(data_scale)

#np.save(f'TSNE_{scaler}', data_embedded)


data_embedded = np.load('TSNE_Quantile.npy')

data_pred = KMeans(n_clusters=10, random_state=random_state).fit_predict(data_embedded)

time_signature = datetime.now().strftime("%m%d-%H%M")
plt.figure(1)
#plt.plot(data_embedded[:,0],data_embedded[:,1],'.')
plt.scatter(data_embedded[:,0],data_embedded[:,1],c=data_pred)
for name in names:
    crit = data_table.loc[:,'SpCl_s_gs_gsu'] == name
    data = data_embedded[crit,:]
    plt.scatter(data[:,0], data[:,1],s=5, label=name)
plt.legend()
plt.savefig(f'TSNE_{time_signature}_{scaler}.pdf')







#print(np.unique(data_table.dtypes))

#print(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))])

#print(data_table.loc[:, (data_table.dtypes == np.float64) |  (data_table.dtypes == np.float64) |(data_table.dtypes == np.int64)])

#data_table = quantile_transform(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))], copy=True)