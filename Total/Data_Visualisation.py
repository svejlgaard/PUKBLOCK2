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
from sklearn.model_selection import StratifiedShuffleSplit

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

class Dataset():
    def __init__(self, filename, filetype, load = True):
        self.filename = filename
        self.filetype = filetype
        if load:
            self.dtable = Table.read(self.filename, format=self.filetype).to_pandas()

    def get_colors(self, all=True, filter=True):
        
        org_names = self.dtable.columns.to_numpy()
        

        if self.filetype == 'fits':

            names = np.array([label for label in org_names if ("mag" in label) | ("photG_g_gs_gsu" in label) | ("e_pmdec_g_gs_gsu" in label)])

            self.obj_names = np.array(self.dtable['Name_u_gsu'],dtype=str)
           
            
            col_indexes = [n for n,name in enumerate(names) if name[0] != "e"]
            col_names = names[col_indexes]
            col_names = [name.replace("mag_s_gs_gsu","") for name in col_names]
            col_names = [name.replace("phot","") for name in col_names]
            col_names = [name.replace("mag_u_gsu","") for name in col_names]
            col_names = [name.replace("mag_w","") for name in col_names]
            col_names = [name.replace("_g_gs_gsu","") for name in col_names]
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


            new_frame = col_data.rename(columns = dict(zip(col_data.columns, col_names)))
            err_frame = err_data.rename(columns = dict(zip(err_data.columns, err_names)))

        elif self.filetype == 'ascii':

            org_data = self.dtable.to_numpy()
            self.obj_names = self.dtable['name']
            self.obj_names = np.array(self.obj_names)
            
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
            err_frame = pd.DataFrame(data=err_data, columns= err_names)

        self.name = ''.join(col_names)

        self.name += f'{self.filetype}'
        
        if filter:
            if self.filetype == 'ascii':
                filt = np.nonzero( (np.abs(err_frame['gerr'].to_numpy()) < 0.5) & 
                                (np.abs(err_frame['rerr'].to_numpy()) < 0.1) & 
                                (np.abs(err_frame['jerr'].to_numpy()) < 0.1) &
                                (np.abs(new_frame['j'].to_numpy() != 0)) &
                                (np.abs(new_frame['k'].to_numpy() != 0)) &
                                (np.abs(new_frame['j'].to_numpy() < 1e5)) &
                                (np.abs(new_frame['k'].to_numpy() < 1e5)) &
                                (np.abs(err_frame['kerr'].to_numpy()) < 0.1) &
                                (np.abs(new_frame['r'].to_numpy() < 19))
                                )
            else:
                filt = np.where( (new_frame['G'] < 20) &
                                (new_frame['r'] < 19)
                        )
            #    filt = np.where(np.isnan(err_frame['W3err']))
                #filt = np.nonzero( (np.abs(err_frame['W3err']) < 1.0) &
                #                 (np.abs(err_frame['W4err'].to_numpy()) < 1.0)
                #                 )
            #    print(len(filt[0]), len(err_frame))
            #    print(abe)
            
            self.dtable = pd.DataFrame(data = self.dtable.loc[filt], columns = org_names)
            
            self.obj_names = self.obj_names[filt]

            self.dtable = self.dtable.reset_index(drop=True)
    
            if type(col_data) == type(np.ones(1)):
                new_frame = pd.DataFrame(data = col_data[filt], columns = col_names)
                err_frame = pd.DataFrame(data = err_data[filt], columns = err_names)
            else:
                
                new_frame = pd.DataFrame(data = col_data.loc[filt].values, columns = col_names)
                err_frame = pd.DataFrame(data = err_data.loc[filt].values, columns = err_names)
                
            
            new_frame = new_frame.reset_index(drop=True)
            err_frame = err_frame.reset_index(drop=True)

        self.color_frame = new_frame
        self.err_frame = err_frame

        assert self.color_frame.shape[0] == self.obj_names.shape[0]

        return new_frame, err_frame

    def get_classes(self):
        if self.filetype == 'fits':
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

        if self.filetype == 'ascii':

            self.labels = np.chararray(self.dtable['qso'].to_numpy().shape, itemsize=3, unicode=True)
            qso_label = np.array(self.dtable['qso'].to_numpy())
            bal_label = np.array(self.dtable['bal'].to_numpy(), dtype=int)
            self.labels[qso_label == 1.0] = str('QSO')
            self.labels[bal_label == 1] = 'BAL'
            self.labels[(qso_label > 1.0) | (qso_label < 1.0)] = 'UNK'
            self.labels = np.array(self.labels)

            self.name += '_for_' + 'qso' + 'bal' + 'unk'
        
        return self.labels





    def remove_color(self, color_x):
        """
        A function to remove a given colorband from the colorframe

        """
        self.color_frame = self.color_frame.drop(columns = color_x)
        try:
            self.err_frame = self.err_frame.drop(columns = f'{color_x}err')
        except:
            print(f'{color_x}err is not found in axis and hence not removed')
        self.name = self.name.replace(f'{color_x}','')
        print(f'Removed {color_x}')
        return self.color_frame, self.err_frame


    def remove_class(self, class_x, some=True):
        if some:
            index_x = np.where(self.labels == class_x)[0]
            chosen = np.random.choice(index_x, size=int(len(index_x)/2))
            self.labels = np.delete(self.labels, chosen)
            self.obj_names = np.delete(self.obj_names, chosen)
            
            self.color_frame = self.color_frame.reset_index(drop=True)
            self.err_frame = self.err_frame.reset_index(drop=True)

            self.color_frame = self.color_frame.drop(index=chosen)
            self.err_frame = self.err_frame.drop(index=chosen)

            self.color_frame = self.color_frame.reset_index(drop=True)
            self.err_frame = self.err_frame.reset_index(drop=True)

            self.name = self.name.replace(f'{class_x}',f'notall{class_x}')
            print(f'Removed some {class_x}')
        else:
            index_not_x = self.labels != class_x
            self.labels = self.labels[index_not_x]
            self.obj_names = self.obj_names[index_not_x]
            self.color_frame = self.color_frame.loc[index_not_x,:]
            self.err_frame = self.err_frame.loc[index_not_x,:]
            self.name = self.name.replace(f'{class_x}','')

            assert self.color_frame.shape[0] == self.obj_names.shape[0]
            print(f'Removed {class_x}')

    
    def color_plot(self, color_x1, color_x2, color_y1, color_y2, save=False):

        x = self.color_frame[color_x1] - self.color_frame[color_x2]

        y = self.color_frame[color_y1] - self.color_frame[color_y2]

        plt.figure(f'{self.filetype}_color')
        for num, l in enumerate(np.unique(self.labels)):
            plt.plot(x[self.labels == l],y[self.labels == l],'.',
                    label=l, 
                    zorder=len(np.unique(self.labels))+1-num,
                    )
        plt.xlabel(f'{color_x1}-{color_x2}')
        plt.ylabel(f'{color_y1}-{color_y2}')
        plt.legend()
        if save:
            plt.savefig(f'plots/{color_x1}{color_x2}{color_y1}{color_y2}_{time_signature}_{self.filetype}.pdf')


    
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
        return self.color_frame, self.labels, self.obj_names, self.scaler


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
            
            print('TSNE-ing')

            if self.filetype == 'fits':
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

            plt.scatter(x[self.qso_condition],y[self.qso_condition], label='QSO')
            plt.scatter(x[self.bal_condition],y[self.bal_condition], label='BAL')
            plt.scatter(x[self.unknown_condition],y[self.unknown_condition], label='UNK')

            if save:
                plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}.pdf')
            
        elif self.filetype == 'fits':
            for num, l in enumerate(np.unique(self.labels)):
                plt.plot(x[self.labels == l],y[self.labels == l],'.', 
                        label=l, 
                        zorder=len(np.unique(self.labels))+1-num,
                        )

            

            plt.legend()

            if with_cluster:
                if len(np.unique(self.labels)) < 3:
                    clusters = len(np.unique(self.labels))
                else:
                    clusters = 6
                #Have tried KMeans, SpectralClustering
                #self.name += 'SC'
                print('Clustering')
                #data_cluster = OPTICS(n_jobs=-1).fit_predict(self.tsne_data)
                data_cluster = SpectralClustering(n_clusters=clusters, 
                                                  random_state=random_state, 
                                                  n_jobs=8,
                                                  ).fit_predict(self.tsne_data)
                plt.figure('Clustering')
                plt.scatter(x,y,c=data_cluster,zorder=10,label='Clusters')
                plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_Clustering.pdf')






class Combination():
    def __init__(self, data_list, label_list, obj_names_list, scaler):
        self.data_list = pd.concat(data_list, sort=True)
        self.color_frame = self.data_list.reset_index(drop=True)
        self.labels = np.concatenate(label_list, axis=0)
        self.obj_names = np.concatenate(obj_names_list)
        self.scaler = scaler

        assert self.color_frame.shape[0] == self.obj_names.shape[0]

        col_names = self.color_frame.columns.to_list()
        self.name = ''.join(col_names)
        self.name += '_Combined'

    def get_classnames(self, save=True):
        name_dataframe = pd.DataFrame(data=[self.obj_names, self.labels])
        name_dataframe = name_dataframe.transpose()
        name_dataframe.columns = ['Name', 'Label']
        self.name_dataframe = name_dataframe
        if save:
            name_dataframe.to_csv('NameFrame.txt', header=1, index=None, sep=' ', mode='a')
    
    def tsne(self, perplexity, save=True, load=False):
        self.labels = self.labels[np.all(self.color_frame.notnull(),axis=1)]
        self.obj_names = self.obj_names[np.all(self.color_frame.notnull(),axis=1)]
        self.color_frame = self.color_frame.dropna()
        self.perplexity = perplexity
        if save:
            data_for_tsne = self.color_frame.copy()
            print('TSNE-ing')

            data_embedded = TSNE(n_components=2,n_jobs=8, perplexity=self.perplexity).fit_transform(data_for_tsne)

            np.save(f'TSNE_{self.scaler}_{self.name}', data_embedded)
            self.tsne_data = data_embedded

        if load:
            try:
                self.tsne_data = np.load(f'TSNE_{self.scaler}_{self.name}.npy')
            except:
                print('No matching TSNE-file! Retry with save=True')
                print(abe)


    def tsne_plot(self, splits, cross = True, save = True, with_cluster = True):
        plt.figure(f'tsne_combination_{self.perplexity}')
        x = self.tsne_data[:,0]
        y = self.tsne_data[:,1]
            
        for num, l in enumerate(np.unique(self.labels)):
            plt.plot(x[self.labels == l],y[self.labels == l],'.', 
                    label=l, 
                    zorder=len(np.unique(self.labels))+1-num,
                    )

        plt.legend()
        if save:
            plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_p{self.perplexity}.pdf')

        if with_cluster:
            clusters = 6
            print('Clustering')
            #if cross:
            #    sss = StratifiedShuffleSplit(n_splits=splits, test_size=0.5, random_state=0)
            #    for train_index, test_index in sss.split(self.tsne_data, self.labels):
            #        data_train, data_test = self.tsne_data[train_index], self.tsne_data[test_index]
            #        labels_train, labels_test = self.labels[train_index], self.labels[test_index]
            SC = SpectralClustering(n_clusters=clusters, 
                                    random_state=random_state, 
                                    n_jobs=8,
                                    )
            self.data_cluster = SC.fit_predict(self.tsne_data)


            plt.figure('Clustering')
            plt.scatter(x,y,c=self.data_cluster)
            if save:
                np.save(f'Clustering_{self.scaler}_{self.name}_p{self.perplexity}', self.data_cluster)
                plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_Clustering.pdf')


    def get_objects(self, save = True, load = True, testing = True):

        if load:
            self.data_cluster = np.load(f'Clustering_{self.scaler}_{self.name}.npy')


        if testing:
            self.data_cluster = np.ones_like(self.obj_names)
            self.data_cluster[::2] = 2
        

        clustered_dict = {}

        for cluster in np.unique(self.data_cluster):
            final_data = self.color_frame[self.data_cluster == cluster]
            final_data = final_data.reset_index(drop=True)
            final_data = final_data.to_numpy()
            final_names = self.obj_names[self.data_cluster == cluster]

            clustered_dict.update({f'Cluster_{cluster}': [final_data, 
                                                         self.obj_names[self.data_cluster == cluster]]})
            
            cluster_frame = pd.DataFrame(data = final_data,
                                         columns = self.color_frame.columns.to_list())

            cluster_frame.insert(0, 'Name', pd.Series(final_names), True)
            
            cluster_frame.to_csv(f'cluster_{cluster}.csv')

        self.clustered_dict = clustered_dict




# Reading the data and converting to a pandas data frame - works more smoothly with sklearn
filename = 'GaiaSDSSUKIDSSAllWISE.fits'

filetype = 'fits'

all_data = Dataset(filename,filetype)

quasar_data = Dataset('MasterCatalogue.dat', 'ascii')

all_data.get_colors(filter=True)

quasar_data.get_colors(filter=True)

all_data.get_classes()

quasar_data.get_classes()


for co in ['u', 'jw', 'hw', 'kw','i', 'W4', 'G', 'W3']:
    all_data.remove_color(co)

for co in ['u', 'i', 'W4', 'W3']:
    quasar_data.remove_color(co)

all_data.color_plot('j','k','g','z', save=False)

quasar_data.color_plot('j','k','g','z', save=False)

all_data_pre, all_classes, all_obj_names, scaler = all_data.preprocess(standard = True, n_quantiles=1000) 

quasar_data_pre, quasar_classes, quasar_obj_names, _ = quasar_data.preprocess(standard = True, n_quantiles=100)



combined_data = Combination([all_data_pre, quasar_data_pre], 
                            [all_classes, quasar_classes], 
                            [all_obj_names, quasar_obj_names], 
                            scaler,
                            )

combined_data.get_classnames(save=True)

perp_list = [50, 75, 100]
split = 5

for p in perp_list:
    combined_data.tsne(p,save=True, load=False)
    combined_data.tsne_plot(split, save=True, with_cluster=True)

combined_data.get_objects(save=True, load=True, testing=False)

