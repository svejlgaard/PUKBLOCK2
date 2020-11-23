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

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

class Dataset():
    def __init__(self, filename, filetype):
        self.filename = filename
        self.filetype = filetype
        self.dtable = Table.read(self.filename, format=self.filetype).to_pandas()


    def get_colors(self, all=True, filter=True):

        org_data = self.dtable.to_numpy()
        org_names = self.dtable.columns.to_numpy()

        if all:
            data = org_data[:,6:-1]
            names = org_names[6:-1]

        col_indexes = [n for n,name in enumerate(names) if ("rr" not in name)]
        col_names = names[col_indexes]
        col_names = [name.replace("AperMag3","") for name in col_names]
        col_data = data[:,col_indexes]

        err_indexes = [n for n,name in enumerate(names) if ("rr" in name)]
        err_names = names[err_indexes]
        err_names = [name.replace("AperMag3","") for name in err_names]
        err_names = [name.replace("E","e") for name in err_names]
        err_data = data[:,err_indexes]


        new_frame = pd.DataFrame(data=col_data, columns=col_names)
        err_frame = pd.DataFrame(data=err_data, columns=err_names)
        #print(err_names)
        if filter:
            filt = np.nonzero( (np.abs(err_frame['gerr'].to_numpy()) < 0.5) & 
                                (np.abs(err_frame['rerr'].to_numpy()) < 0.1) & 
                                (np.abs(err_frame['jerr'].to_numpy()) < 0.1) & 
                                (np.abs(err_frame['kerr'].to_numpy()) < 0.1) &
                                (np.abs(err_frame['uerr'].to_numpy()) < 0.5) &
                                (np.abs(err_frame['ierr'].to_numpy()) < 0.5) &
                                (np.abs(err_frame['zerr'].to_numpy()) < 0.5) &
                                (np.abs(err_frame['yerr'].to_numpy()) < 0.5) &
                                (np.abs(err_frame['herr'].to_numpy()) < 0.5) 
                                )

        self.dtable = pd.DataFrame(data = org_data[filt], columns = org_names)
            
        new_frame = pd.DataFrame(data = col_data[filt], columns = col_names)
        err_frame = pd.DataFrame(data = err_data[filt], columns = err_names)

        
        self.color_frame = new_frame
        self.err_frame = err_frame
        return new_frame, err_frame
    
    def color_plot(self, color_x1, color_x2, color_y1, color_y2, save=False):
        
        x = self.color_frame[color_x1].to_numpy() - self.color_frame[color_x2].to_numpy()
        
        y = self.color_frame[color_y1] - self.color_frame[color_y2]

        qso_label = self.dtable['qso']

        self.qso_condition = (x!= 0) & (qso_label == 1)

        bal_label = self.dtable['bal']

        self.bal_condition = (x!= 0) & (bal_label == 1)

        self.unknown_condition = (x!= 0) & (qso_label > 1)

        colormap = ['r', 'g', 'b']
        plt.figure(1)
        plt.plot(x[self.qso_condition],y[self.qso_condition],'.', label='QSO', color=colormap[0])
        plt.plot(x[self.bal_condition],y[self.bal_condition],'.',label= 'BAL', color=colormap[1])
        plt.plot(x[self.unknown_condition], y[self.unknown_condition],'.', label='UNK', color=colormap[2])
        plt.xlabel(f'{color_x1}-{color_x2}')
        plt.ylabel(f'{color_y1}-{color_y2}')
        plt.legend()

        if save:
            plt.savefig(f'plots/{color_x1}{color_x2}{color_y1}{color_y2}_{time_signature}.pdf')



# Reading the data and converting to a pandas data frame - works more smoothly with sklearn
filename = 'MasterCatalogue.dat'

filetype = 'ascii'

quasars = Dataset(filename,filetype)
color_frame, err_frame = quasars.get_colors()

quasars.color_plot('j','k','W1','W2')



# Creating a copy for the preprocessing

data_scale = color_frame.to_numpy()


# Preprocessing the data (not any data of string-type) via the quantile transformer


scaler = StandardScaler(copy=True)
data_scale = scaler.fit_transform(data_scale)

#min_max_scaler = MinMaxScaler()

#data_scale = min_max_scaler.fit_transform(data_scale)


scaler = 'Quantile'

data_scale = quantile_transform(data_scale, copy=True, n_quantiles=300)


print('TSNE-ing')
# Clustering the data via t-SNE

data_embedded = TSNE(n_components=2,n_jobs=-1).fit_transform(data_scale)

#np.save(f'TSNE_{scaler}', data_embedded)


#data_embedded = np.load('TSNE_Quantile.npy')

#data_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(data_embedded)

plt.figure(2)
x = data_embedded[:,0]
y = data_embedded[:,1]
plt.scatter(x[quasars.qso_condition],y[quasars.qso_condition], label='QSO', c='r')
plt.scatter(x[quasars.bal_condition],y[quasars.bal_condition], label='BAL', c='g')
plt.scatter(x[quasars.unknown_condition],y[quasars.unknown_condition], label='UNK', c='b')
plt.legend()
plt.savefig(f'plots/TSNE_{time_signature}_{scaler}.pdf')







#print(np.unique(data_table.dtypes))

#print(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))])

#print(data_table.loc[:, (data_table.dtypes == np.float64) |  (data_table.dtypes == np.float64) |(data_table.dtypes == np.int64)])

#data_table = quantile_transform(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))], copy=True)