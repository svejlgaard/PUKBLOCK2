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
from sklearn.cluster import KMeans, SpectralClustering, OPTICS, AgglomerativeClustering
from sklearn.model_selection import StratifiedShuffleSplit

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

class Dataset():


    def __init__(self, filename, filetype, magnitude):
        """
        A class for loading and preprocessing individual files.

        Parameters:

        filename: str, name of the file to be loaded, should be in the same folder as this script
        filetype: str, type of the file to be loaded, only supported for [fits, ascii]
        magnitude: int, wanted maximum for the GAIA G-filter magnitude

        """

        self.filename = filename
        self.filetype = filetype
        self.magnitude = magnitude
        self.dtable = Table.read(self.filename, format=self.filetype).to_pandas()


    def get_colors(self, filter=True):
        """
        A function for extracting the photometric data in the given file and save the name of the file.
        Assuming the photometric data includes 'mag' in the label. The GAIA G label should include 'photG_g_gs_gsu'.

        Parameters:

        filter: bool, if True ascii files will only include objects with G smaller than the wanted magnitude and r < 19, while
        fits files will only include objects with G smaller than the wanted magnitude, r < 19 and several constraints on the associated errors.

        Returns:

        new_frame: pandas DataFrame, the photometric data with labels
        err_frame: pandas DataFrame, the associated error on the photemetric data with labels

        """
        
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
        self.name += f'{self.magnitude}'
        
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
                filt = np.where( (new_frame['G'] < self.magnitude) & 
                                 (new_frame['r'] < 19))
            
            self.dtable = pd.DataFrame(data = self.dtable.loc[filt], columns = org_names)
            self.obj_names = self.obj_names[filt]
            self.dtable = self.dtable.reset_index(drop=True)
    
            # Handling numpy array types and pandas dataframe types seperately
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
        """
        A function for extracting the GAIA classification labels for the objects in the file. 
        Assuming the classification label includes 'SpCl_s_gs_gsu' for fits files.

        Returns:

        labels: numpy array, including either ['UNK', 'QSO', 'GALAXY', 'STAR', 'BAL'] for each object

        """

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
        A function for removing photometric data as part of the preprocessing. 
        Includes the possibility that the photometric data is not associated with an error.

        Parameters:

        color_x: str, the label name of the photometric filter to be removed

        Returns:

        color_frame: pandas DataFrame, the new photometric data with labels 
        err_frame: pandas DataFrame, the new associated error on the photemetric data with labels

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
        """
        A function for removing a specified class as part of the preprocessing. 
        Includes the possibility to remove only some objects within this class as part of testing.

        Parameters:

        class_x: str, the label name of the object to be removed
        """
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
        """
        A function for saving a color-color plot for color_x1-color_x2 vs color_y1-color_y2.

        Parameters:

        color_x1: str, the positive color on the x-axis
        color_x2: str, the negative color on the x-axis
        color_y1: str, the positive color on the y-axis
        color_y2: str, the negative color on the y-axis
        save: if True the plot is saved as a pdf in a subfolder called 'plots'

        Returns:

        plt.figure: pdf called color_x1-color_x2color_y1-colory2_thecurrenttime_thegivenfiletype in subfolder called 'plots' 

        """
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
        """
        A function for standardizing the data set as the final part of the preprocessing.

        Parameters:

        standard: bool, if True sklearn.StandardScaler is applied to the data set
        quantile: bool, if True sklearn.quantile_transform is applied to the data with quantiles as given by n_quantiles
        n_quantiles: int, the number of qunatiles to use for the quantile transformer
        
        Returns:

        color_frame: pandas DataFrame, the standardized photometric data set
        labels: numpy array, including either ['UNK', 'QSO', 'GALAXY', 'STAR', 'BAL'] for each object
        obj_names: numpy array, includer the name of each object
        scaler: str, names of the applied preprocessing algorithms

        """
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




class Combination():
    def __init__(self, magnitude, data_list, label_list, obj_names_list, scaler):
        """
        A class for analysing a list of files to be combined.

        Parameters:

        magnitude: int, wanted maximum for the GAIA G-filter magnitude
        data_list: list, a list of pandas DataFrames to be combined and analysed
        label_list: list, a list of numpy arrays associated to the given data_list
        obj_names_list: list, a list of numpy str-type arrays associated to the given data_list
        scaler: str, names of the applied preprocessing algorithms

        """
        self.data_list = pd.concat(data_list, sort=True)
        self.color_frame = self.data_list.reset_index(drop=True)
        self.labels = np.concatenate(label_list, axis=0)
        self.obj_names = np.concatenate(obj_names_list)
        self.scaler = scaler
        self.magnitude = magnitude
        assert self.color_frame.shape[0] == self.obj_names.shape[0]
        col_names = self.color_frame.columns.to_list()
        self.name = ''.join(col_names)
        self.name += '_Combined'


    def get_classnames(self, save=True):
        """
        A function for saving the label and name of each object.

        Parameters:

        save: bool, if True a pandas DataFrame including the names and labels of all objects is saved in a txt file

        """
        name_dataframe = pd.DataFrame(data=[self.obj_names, self.labels])
        name_dataframe = name_dataframe.transpose()
        name_dataframe.columns = ['Name', 'Label']
        self.name_dataframe = name_dataframe
        if save:
            name_dataframe.to_csv('NameFrame.txt', header=1, index=None, sep=' ')
    

    def tsne(self, perplexity, save=True, load=False):
        """
        A function for applied the t-SNE algorithm to the combined data set.

        Parameters:

        perplexity: int, value of the perplexity parameter given to the t-SNE algorithm
        save: bool, if True the transformed data from the t-SNE algorithm is saved as a npy file

        Returns:

        np.save: if save, a file called TSNE_currenttime_usedscaler_nameofdatafile_pperplexity.npy in the current folder

        """
        self.labels = self.labels[np.all(self.color_frame.notnull(),axis=1)]
        self.obj_names = self.obj_names[np.all(self.color_frame.notnull(),axis=1)]
        self.color_frame = self.color_frame.dropna()
        self.perplexity = perplexity
        self.name += f'_p{self.perplexity}'
        data_for_tsne = self.color_frame.copy()
        print('TSNE-ing')
        data_embedded = TSNE(n_components=2,n_jobs=8, perplexity=self.perplexity).fit_transform(data_for_tsne)
        self.tsne_data = data_embedded
        if save:
            np.save(f'TSNE_{time_signature}_{self.scaler}_{self.name}_p{self.perplexity}', data_embedded)


    def tsne_plot(self, save = True, with_cluster = True):
        """
        A function for plotting and clustering the data from the t-SNE algorithm to the combined data set.

        Parameters:

        save: bool, if True the transformed data from the t-SNE algorithm is plotted and saved
        with_cluster: bool, if True the t-SNE transformed data is divided into clusters via the sklearn.ApplomerativeClustering algorithm, 
                      plotted and saved as a npy-file if save = True

        """
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
            plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_G{self.magnitude}.pdf')
            print(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_G{self.magnitude}.pdf')

        if with_cluster:
            clusters = int(input('Number of clusters: '))
            AC = AgglomerativeClustering(n_clusters=clusters, linkage='ward')
            self.data_cluster = AC.fit_predict(self.tsne_data)
            plt.figure(f'ClusteringP{self.perplexity}')
            plt.scatter(x,y,c=self.data_cluster)
            if save:
                np.save(f'Clustering_{self.scaler}_{self.name}_p{self.perplexity}', self.data_cluster)
                print(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_G{self.magnitude}_Clustering.pdf')
                plt.savefig(f'plots/TSNE_{time_signature}_{self.scaler}_{self.name}_G{self.magnitude}_Clustering.pdf')


    def get_objects(self, save = True, load = True):
        """
        A function for saving the objects according to their given cluster.

        Parameters:

        load: bool, if True load the npy-file with the clustered data, must be set to True if run without tsne_plot
        save: bool, if True save the resulting clusters in seperate csv-files

        Returns:

        clustered_dict: dictionary, a pandas DataFrame for each entry given by the clusters 
        """
        if load:
            self.data_cluster = np.load(f'Clustering_{self.scaler}_{self.name}_p{self.perplexity}.npy')

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
            print(f'cluster_{cluster}_p{self.perplexity}_G{self.magnitude}.csv')
            cluster_frame.to_csv(f'cluster_{time_signature}_{cluster}_p{self.perplexity}_G{self.magnitude}.csv')
        self.clustered_dict = clustered_dict
        return clustered_dict



# How the classes have been used in this project
M = 20

# This file should be found elsewhere as it is too large for github
filename = 'GaiaSDSSUKIDSSAllWISE.fits'

filetype = 'fits'

all_data = Dataset(filename,filetype, magnitude=M)

quasar_data = Dataset('MasterCatalogue.dat', 'ascii', magnitude=M)

all_data.get_colors(filter=True)

quasar_data.get_colors(filter=True)

all_data.get_classes()

quasar_data.get_classes()


for co in ['u', 'jw', 'hw', 'kw','i', 'W4', 'G', 'W3']:
    all_data.remove_color(co)

for co in ['u', 'i', 'W4', 'W3']:
    quasar_data.remove_color(co)

quasar_data.remove_class('UNK', some=False)

all_data.color_plot('j','k','g','z', save=False)

quasar_data.color_plot('j','k','g','z', save=False)

all_data_pre, all_classes, all_obj_names, scaler = all_data.preprocess(standard = True, n_quantiles=1000) 

quasar_data_pre, quasar_classes, quasar_obj_names, _ = quasar_data.preprocess(standard = True, n_quantiles=100)



perp_list = [90]

for p in perp_list:
    combined_data = Combination(M,[all_data_pre, quasar_data_pre], 
                            [all_classes, quasar_classes], 
                            [all_obj_names, quasar_obj_names], 
                            scaler,
                            )
    combined_data.get_classnames(save=True)
    combined_data.tsne(p, save=True)
    combined_data.tsne_plot(save=True, with_cluster=True)
    combined_data.get_objects(save=True, load=True)

