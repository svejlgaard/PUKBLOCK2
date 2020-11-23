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
from sklearn.cluster import KMeans, SpectralClustering, OPTICS

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
        
        org_names = self.dtable.columns.to_numpy()

        if self.filetype == 'fits':

            #| (label == "SpCl_s_gs_gsu")

            names = np.array([label for label in org_names if ("mag" in label)])


            col_indexes = [n for n,name in enumerate(names) if name[0] != "e"]
            col_names = names[col_indexes]
            col_names = [name.replace("mag_s_gs_gsu","") for name in col_names]
            col_names = [name.replace("mag_u_gsu","") for name in col_names]
            col_names = [name.replace("mag_w","") for name in col_names]
            col_names = [name.replace("Y","y") for name in col_names]
            col_names = [name.replace("J","j") for name in col_names]
            col_names = [name.replace("H","h") for name in col_names]
            col_names = [name.replace("K","k") for name in col_names]
            for n in range(len(col_names)-3,len(col_names)):
                col_names[n] += "w"
            col_data = self.dtable.loc[:,names[col_indexes]]
            
            

            err_indexes = [n for n,name in enumerate(names) if name[0] == "e"]
            err_names = [name + "err" for name in col_names]
            err_data = self.dtable.loc[:,names[err_indexes]]

        elif self.filetype == 'ascii':

            org_data = self.dtable.to_numpy()

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

        new_frame = col_data.rename(columns = dict(zip(col_data.columns, col_names)))
        err_frame = err_data.rename(columns = dict(zip(err_data.columns, err_names)))

        self.name = ''.join(col_names)
        
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
            self.dtable = pd.DataFrame(data = self.dtable.loc[filt], columns = org_names)
            
            new_frame = pd.DataFrame(data = col_data[filt], columns = col_names)
            err_frame = pd.DataFrame(data = err_data[filt], columns = err_names)

        
        self.color_frame = new_frame
        self.err_frame = err_frame

        return new_frame, err_frame

    def get_classes(self):
        if filetype == 'fits':
            labels = self.dtable["SpCl_s_gs_gsu"]
            labels = [str(n) for n in labels]
            labels = [n.replace('b','') for n in labels]
            labels = [n.replace(' ','') for n in labels]
            labels = [n.replace("'", "") for n in labels]
            for n in range(len(labels)):
                if labels[n] == '':
                    labels[n] += 'UNK'  
            self.labels = np.array(labels)
            self.dtable["SpCl_s_gs_gsu"] = labels  
            self.name += '_for_' + ''.join(np.unique(labels))

        elif filetype == 'ascii':
            qso_label = self.dtable['qso']

            self.qso_condition = qso_label == 1

            bal_label = self.dtable['bal']

            self.bal_condition = bal_label == 1

            self.unknown_condition = qso_label > 1

            self.name += '_for_' + 'qso' + 'bal' + 'unk'


    def remove_color(self, color_x):
        """
        A function to remove a given colorband from the colorframe

        """
        self.color_frame = self.color_frame.drop(columns = color_x)
        self.err_frame = self.err_frame.drop(columns = f'{color_x}err')
        self.name = self.name.replace(f'{color_x}','')
        print(f'Removed {color_x}')
        return self.color_frame, self.err_frame


    def remove_class(self, class_x):

        if self.filetype == 'fits':
            index_not_x = self.labels != class_x
            self.labels = self.labels[index_not_x]
            self.color_frame = self.color_frame.loc[index_not_x,:]
            self.err_frame = self.err_frame.loc[index_not_x,:]
            self.name = self.name.replace(f'{class_x}','')
            print(f'Removed {class_x}')
        

        elif self.filetype == 'ascii':
            print('NOT IMPLEMENTED!')
            print(abe)
    
    def color_plot(self, color_x1, color_x2, color_y1, color_y2, save=False):

        if self.filetype == 'fits':
            x = self.color_frame[color_x1] - self.color_frame[color_x2]

            y = self.color_frame[color_y1] - self.color_frame[color_y2]

            colormap = ['r', 'g', 'b','y']
            plt.figure(1)
            for num, l in enumerate(np.unique(self.labels)):
                plt.plot(x[self.labels == l],y[self.labels == l],'.',
                        label=l, color=colormap[num], 
                        zorder=len(np.unique(self.labels))+1-num,
                        )
            plt.xlabel(f'{color_x1}-{color_x2}')
            plt.ylabel(f'{color_y1}-{color_y2}')
            plt.legend()
            

        elif self.filetype == 'ascii':
        
            x = self.color_frame[color_x1].to_numpy() - self.color_frame[color_x2].to_numpy()
            

            y = self.color_frame[color_y1] - self.color_frame[color_y2]

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

    
    def preprocess(self, standard = True, quantile = True, n_quantiles = 1000):
        self.scaler = ''
        if standard:
            self.scaler += 'SS'
            scaler = StandardScaler(copy=True)
            self.color_frame = pd.DataFrame(scaler.fit_transform(self.color_frame), columns = self.color_frame.columns)
        if quantile:
            self.scaler += 'Quantile'
            self.color_frame = pd.DataFrame(quantile_transform(self.color_frame, copy=True, n_quantiles=n_quantiles),
                                            columns = self.color_frame.columns,
                                            )


    def tsne(self, save = True, load = False, diff = False):
        if filetype == 'fits':
            self.labels = self.labels[np.all(self.color_frame.notnull(),axis=1)]

        elif filetype == 'ascii':
            self.bal_condition = self.bal_condition[np.all(self.color_frame.notnull(),axis=1)]
            self.qso_condition = self.qso_condition[np.all(self.color_frame.notnull(),axis=1)]
            self.unknown_condition = self.unknown_condition[np.all(self.color_frame.notnull(),axis=1)]

        if save:
            data_for_tsne = self.color_frame.dropna()
            if diff:
                new_data_for_tsne = pd.DataFrame(np.zeros_like(data_for_tsne),columns=data_for_tsne.columns)
                for i in range(data_for_tsne.shape[1]):
                    more_data_for_tsne = data_for_tsne.diff(axis=1)
                    shifted_names = np.roll(more_data_for_tsne.columns.to_list(),shift=(i+1))
                    new_names = list(map('-'.join, zip(data_for_tsne.columns.to_list(), shifted_names)))
                    more_data_for_tsne = more_data_for_tsne.rename(columns = dict(zip(more_data_for_tsne.columns, new_names)))
                    more_data_for_tsne = more_data_for_tsne.dropna(axis=1)
                    more_data_for_tsne = more_data_for_tsne.reset_index(drop=True)
                    new_data_for_tsne = pd.concat([new_data_for_tsne, more_data_for_tsne],axis=1)
                new_data_for_tsne = new_data_for_tsne.loc[:, (new_data_for_tsne != 0).any(axis=0)]
                print(new_data_for_tsne)
            print('TSNE-ing')
            data_embedded = TSNE(n_components=2,n_jobs=8).fit_transform(data_for_tsne)

            np.save(f'TSNE_{self.scaler}_{self.name}', data_embedded)
            self.tsne_data = data_embedded

        if load:
            try:
                self.tsne_data = np.load(f'TSNE_{self.scaler}_{self.name}.npy')
            except:
                print('No matching TSNE-file! Retry save=True')
                print(abe)
        
    def tsne_plot(self, save = True, with_cluster = True):
        plt.figure(2)
        x = self.tsne_data[:,0]
        y = self.tsne_data[:,1]
        if self.filetype == 'ascii':
            plt.scatter(x[self.qso_condition],y[self.qso_condition], label='QSO', c='r')
            plt.scatter(x[self.bal_condition],y[self.bal_condition], label='BAL', c='g')
            plt.scatter(x[self.unknown_condition],y[self.unknown_condition], label='UNK', c='b')
            
        elif self.filetype == 'fits':
            colormap = ['r', 'g', 'b', 'y']
            for num, l in enumerate(np.unique(self.labels)):
                plt.plot(x[self.labels == l],y[self.labels == l],'.', 
                        label=l, color=colormap[num], 
                        zorder=len(np.unique(self.labels))+1-num,
                        )

            
            if with_cluster:
                if len(np.unique(self.labels)) < 3:
                    clusters = len(np.unique(self.labels))
                else:
                    clusters = 3
                #Have tried KMeans, SpectralClustering
                self.name += 'SC'
                print(self.tsne_data.shape)
                print('Clustering')
                #data_cluster = OPTICS(n_jobs=-1).fit_predict(self.tsne_data)
                data_cluster = SpectralClustering(n_clusters=clusters, 
                                                  random_state=random_state, 
                                                  n_jobs=8,
                                                  ).fit_predict(self.tsne_data)
                plt.scatter(x,y,c=data_cluster,zorder=10,label='Clusters')


        
        plt.legend()
        if save:
            plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}.pdf')




# Reading the data and converting to a pandas data frame - works more smoothly with sklearn
filename = 'GaiaSDSSUKIDSSAllWISE.fits'

filetype = 'fits'

all_data = Dataset(filename,filetype)

all_data.get_colors(filter=False)

all_data.get_classes()

for co in ['u', 'jw', 'hw', 'kw','W3','W4','i']:
    all_data.remove_color(co)

for cl in ['GALAXY']:
    all_data.remove_class(cl)

all_data.color_plot('j','k','g','z', save=False)

all_data.preprocess(standard = True)

all_data.tsne(save=True, load=False, diff=False)

all_data.tsne_plot(save=True, with_cluster=True)

# Preprocessing the data (not any data of string-type) via the quantile transformer




#data_embedded = np.load('TSNE_Quantile.npy')

#data_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(data_embedded)







#print(np.unique(data_table.dtypes))

#print(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))])

#print(data_table.loc[:, (data_table.dtypes == np.float64) |  (data_table.dtypes == np.float64) |(data_table.dtypes == np.int64)])

#data_table = quantile_transform(data_table.loc[:, data_table.dtypes == np.any(np.unique(data_table.dtypes))], copy=True)