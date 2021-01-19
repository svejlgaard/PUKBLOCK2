import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import ascii
import os, contextlib, sys
from datetime import datetime
from glob import glob
import csv
from tqdm import tqdm
from PyAstronomy import pyasl
from astropy import coordinates as coords
from astroquery.sdss import SDSS
import re


# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']

class PlotElm():
    def __init__(self, gvalue, nclusters, pvalue):
    """
    A class for plotting the results.

    Parameters:

    gvalue: The upper bound on the GAIA G
    nclusters: The number of clusters found by the AgglomerativeClustering
    pvalue: The perplexity value used in t-SNE
    """
        self.gvalue = gvalue
        self.nclusters = nclusters
        self.pvalue = pvalue
        os.chdir(sys.path[0])
        self.files = glob(f'Total/G{self.gvalue}/final/*_p{self.pvalue}_G{self.gvalue}_labels_clean_fewerunks.csv')
    
    def colorplot(self, labellist, colorplus, colorminus, save=True):
        """
        A function for saving a histogram of colorplus-colorminus.

        Parameters:

        colorplus: str, the positive color on the x-axis
        colorminus: str, the negative color on the x-axis
        save: if True the plot is saved as a pdf in a subfolder called 'plots'

        Returns:

        plt.figure: pdf called Colorplot_colorpluscolorminus in subfolder called 'Report/figures' 

        """
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        for file in self.files:
            print(file)
            name = file.split('_')[2]
            data = pd.read_csv(file)
            data = data.loc[(data['Label'] == labellist[0]) | (data['Label'] == labellist[1])]
            data_plus = data[colorplus].to_numpy()
            data_minus = data[colorminus].to_numpy()
            if len(data_plus) > 0: 
                ax.hist(data_plus-data_minus, 50, 
                        histtype='step',
                        density=True, 
                        linewidth = 3,
                        label = f'Cluster {name} with mean {np.mean(data_plus-data_minus):.2f}')
        ax.set(xlabel = f'{colorplus} - {colorminus}', ylabel = 'Density')
        plt.legend()
        if save:
            plt.savefig(f'Report/figures/Colorplot_{colorplus}{colorminus}.pdf')
    
    def meanplot(self, labellist, feature, save=True):
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        for file in self.files:
            print(file)
            name = file.split('_')[2]
            data = pd.read_csv(file)
            data = data.loc[(data['Label'] == labellist[0]) | (data['Label'] == labellist[1])]
            chosen = data[feature].to_numpy()
            chosen = [val.replace('[','') for val in chosen]
            chosen = [val.replace(']','') for val in chosen]
            chosen = [val for val in chosen if val]
            chosen = np.array(chosen, dtype=float)
            if (feature == 'z') or (feature == 'AV'):
                chosen = chosen[chosen < 50]
            if len(chosen) > 0:
                print(f'Number of objects in cluster {name}: {len(chosen)}')
                mean_values = list()
                for k in tqdm(range(100000)):
                    indices = np.random.randint(0, len(chosen)-1, size = len(chosen))
                    sample = chosen[indices]
                    sample_mean = np.mean(sample)
                    mean_values.append(sample_mean)
                lower = np.percentile(mean_values, 2.5)
                upper = np.percentile(mean_values, 97.5)
                print(f'The confidence interval in the mean of cluster {name} is: {lower, upper}')
                
                ax.hist(chosen, 20, 
                        histtype='step',
                        density=True, 
                        linewidth = 3,
                        label = f'Cluster {name} with {feature} mean {np.mean(chosen):.2f} and CI [{lower:.2f}, {upper:.2f}]')
        ax.set(xlabel = f'{feature}', ylabel = 'Density')
        plt.legend()
        if save:
            plt.savefig(f'Report/figures/Featureplot_{feature}.pdf')

    def circleplot(self, outlier):
        for file in self.files:
            fig = plt.figure(figsize=(9, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            fig.subplots_adjust(wspace=-1.2)

            name = file.split('_')[2]

            data = pd.read_csv(file)
            bals = data.loc[data['Label'] == 'BAL']
            bals['Subclass'] = 'bal'
            bals['Label'] = 'QSO'
            data.loc[data['Label'] == 'BAL'] = bals
            data.loc[data['Label'] == ''] = 'GALAXY'


            gals = data.loc[data['Label'] == 'GALA']
            gals['Label'] = 'GALAXY'
            data.loc[data['Label'] == 'GALA'] = gals


            labels = np.unique(data['Label'])
            labels = labels[(labels != 'UNK') & (labels != 'UNKN') & (labels != '[]')]
            
            if outlier not in labels:
                continue
            outlier_arg = np.argwhere(labels == outlier)[0][0]
            labels = labels.tolist()
            
            labels.remove(outlier)
            labels.insert(0, outlier)
            
            labels = np.array(labels)
            if len(labels) > 0:
                sizes = np.ones_like(labels, dtype=float)
                col_subsizes = np.ones_like(labels).tolist()
                col_subclasses = np.ones_like(labels).tolist()
                
                for i in range(len(labels)):
                    i_data = data.loc[data['Label']==labels[i]]
                    sizes[i] = len(i_data)
                    smallclasses = i_data['Subclass'].to_numpy(dtype=str)
                    for p, q in enumerate(smallclasses):
                        if q[0] == 's':
                            q = re.sub(r'\d+', '', q)
                            smallclasses[p] = q
                        if 'broadline' in q:
                            q = q.replace('broadline', 'bl')
                            smallclasses[p] = q
                        if 'starburst' in q:
                            q = q.replace('starburst', 'sb')
                            smallclasses[p] = q
                        if 'starforming' in q:
                            q = q.replace('starforming', 'sf')
                            smallclasses[p] = q

                    if (labels[i] == 'QSO') | (labels[i] == 'BAL'):
                        smallclasses[smallclasses == 'nan'] = 'qso' 
                    if (labels[i] == 'STAR') | (labels[i] == 'GALAXY'):
                        smallclasses[smallclasses == 'nan'] = 'unk'
                    smallclasses[smallclasses == 'star calciumwd'] = 'white dwarf'
                    subclasses = np.unique(smallclasses)
                    col_subclasses[i] = subclasses
                    subsizes = np.ones_like(subclasses, dtype=int)
                    for j in range(len(subclasses)):
                        subsizes[j] = len(i_data.loc[smallclasses==subclasses[j]])
                    col_subsizes[i] = subsizes
                dom_label = labels[np.argmax(sizes)]
                outlier_arg = np.argwhere(labels == outlier)[0][0]
                explode = np.zeros_like(labels, dtype=float)
                explode[labels == outlier] = 0.1

                all_colorlist = ['forestgreen', 'firebrick', 'royalblue','rebeccapurple']
                angle = -45
                ratios = sizes / np.sum(sizes)
                perc = ratios * 100
                figlabels = [l + f': ~ {perc[m]:.1f} %' for m, l in enumerate(labels)]
                ax1.pie(sizes, labels=figlabels, explode=explode, colors=all_colorlist, startangle=float(angle), radius = 0.9)
                ax1.set_title('Distribution in cluster')
                # bar chart parameters
                xpos = 0
                bottom = 0
                ratios = col_subsizes[0]
                ratios = ratios / sum(ratios)
                width = .15
                cmap = plt.cm.get_cmap('Greens')
                ns = np.linspace(0.2,1, num=len(ratios))
                for j in range(len(ratios)):
                    height = ratios[j]
                    rgba = cmap(ns[j])
                    ax2.bar(xpos, height, width, bottom=bottom, color=rgba)
                    ypos = bottom + ax2.patches[j].get_height() / 2
                    bottom += height

                ax2.set_title(f'Subclasses in {outlier}')
                legend = col_subclasses[outlier_arg]
                perc = ratios * 100
                fig2labels = [l + f': ~ {perc[m]:.1f} %' for m, l in enumerate(legend)]
                
                ax2.legend((fig2labels), loc='center right')
                ax2.axis('off')
                ax2.set_xlim(- 2.5 * width, 2.5 * width)
                plt.tight_layout()
                plt.savefig(f'Report/figures/PieChart_cluster{name}_p{self.pvalue}_G{self.gvalue}_{outlier}.pdf')
                fig.clf()

    def comparison(self, obj):
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        for file in self.files:
            print(file)
            name = file.split('_')[1]
            data = pd.read_csv(file)
            labels = np.unique(data['Label'])
            labels = labels[(labels != 'UNK') & (labels != 'UNKN')]
            if len(labels) > 0:
                sizes = np.ones_like(labels)
                for i in range(len(labels)):
                    sizes[i] = len(data.loc[data['Label']==labels[i]])
                dom_label = labels[np.argmax(sizes)]
                if obj != dom_label:
                    outliers = data.loc[data['Label'] == obj]
                    outliers.reset_index(drop=True)
                    outliers.to_csv(f'Report/figures/{obj}_outliers_in_cluster_{name}_p{self.pvalue}_G{self.gvalue}.csv')

    def combined(self):
        data_list = list()
        for file in self.files:
            data = pd.read_csv(file)
            print(data.shape)
            data_list.append(data)
        data = pd.concat(data_list)

        print('\n')
        print(f'The total number of datapoints in G < {self.gvalue} is {data.shape[0]}')
        print('\n')

        bals = data.loc[data['Label'] == 'BAL']
        bals['Subclass'] = 'bal'
        bals['Label'] = 'QSO'
        data.loc[data['Label'] == 'BAL'] = bals
        data.loc[data['Label'] == ''] = 'GALAXY'
        gals = data.loc[data['Label'] == 'GALA']
        gals['Label'] = 'GALAXY'
        data.loc[data['Label'] == 'GALA'] = gals
        labels = np.unique(data['Label'])
        labels = labels[(labels != 'UNK') & (labels != 'UNKN') & (labels != '[]')]
    
        if len(labels) > 0:
            sizes = np.ones_like(labels, dtype=float)
            col_subsizes = np.ones_like(labels).tolist()
            col_subclasses = np.ones_like(labels).tolist()
            for i in range(len(labels)):
                i_data = data.loc[data['Label']==labels[i]]
                sizes[i] = len(i_data)
                smallclasses = i_data['Subclass'].to_numpy(dtype=str)
                for p, q in enumerate(smallclasses):
                    if q[0] == 's':
                        q = re.sub(r'\d+', '', q)
                        smallclasses[p] = q
                    if 'broadline' in q:
                        q = q.replace('broadline', 'bl')
                        smallclasses[p] = q
                    if 'starburst' in q:
                        q = q.replace('starburst', 'sb')
                        smallclasses[p] = q
                    if 'starforming' in q:
                        q = q.replace('starforming', 'sf')
                        smallclasses[p] = q

                if (labels[i] == 'QSO') | (labels[i] == 'BAL'):
                    smallclasses[smallclasses == 'nan'] = 'qso' 
                 
                if (labels[i] == 'STAR') | (labels[i] == 'GALAXY'):
                    smallclasses[smallclasses == 'nan'] = 'unk'
                smallclasses[smallclasses == 'star calciumwd'] = 'white dwarf'
                subclasses = np.unique(smallclasses)
                col_subclasses[i] = subclasses
                subsizes = np.ones_like(subclasses, dtype=int)
                for j in range(len(subclasses)):
                    subsizes[j] = len(i_data.loc[smallclasses==subclasses[j]])
                col_subsizes[i] = subsizes

            plt.figure(5)
            plt.clf()
            all_colorlist = ['forestgreen', 'firebrick', 'royalblue','rebeccapurple']
            angle = -45
            ratios = sizes / np.sum(sizes)
            perc = ratios * 100
            figlabels = [l + f': ~ {perc[m]:.1f} %' for m, l in enumerate(labels)]
            plt.pie(sizes, labels=figlabels, colors=all_colorlist, startangle=float(angle), radius = 1)
            plt.title(f'Distribution for G < {self.gvalue}')
            plt.savefig(f'Report/figures//PieChart_all_p{self.pvalue}_G{self.gvalue}.pdf')

            plt.figure(6)

            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,9))
            # bar chart parameters
            colorlist = ['Greens', 'Reds', 'Blues']
            legendlist = list()
            for i in range(len(labels)):
                xpos = 0
                bottom = 0
                ratios = col_subsizes[i]
                ratios = ratios / sum(ratios)
                width = .15
                cmap = plt.cm.get_cmap(colorlist[i])
                ns = np.linspace(0.2,1, num=len(ratios))
                for j in range(len(ratios)):
                    height = ratios[j]
                    rgba = cmap(ns[j])
                    axs[i].bar(xpos, height, width, bottom=bottom, color=rgba)
                    ypos = bottom + axs[i].patches[j].get_height() / 2
                    bottom += height

                axs[i].set_title(f'{labels[i]}', size = 30)
                legend = col_subclasses[i]
                perc = ratios * 100
                fig2labels = [l + f': ~ {perc[m]:.1f} %' for m, l in enumerate(legend)]
                legendlist.append(fig2labels)
                axs[i].axis('off')
                axs[i].set_xlim(- 2.5 * width, 2.5 * width)
            for n, ax in enumerate(axs):
                ax.legend(legendlist[n],loc='center left', prop={'size': 11})

            plt.tight_layout()
            plt.savefig(f'Report/figures/PieChart_subclasses_all_p{self.pvalue}_G{self.gvalue}.pdf')
            fig.clf()


# Example of usage, requires figures not found on github

brightest = PlotElm(18, 2, 20)
brightest.circleplot('STAR')
brightest.circleplot('QSO')
brightest.circleplot('GALAXY')
brightest.combined()


# clearer = PlotElm(19, 2, 50)
# clearer.circleplot('STAR')
# clearer.circleplot('QSO')
# clearer.circleplot('GALAXY')
# clearer.combined()


# faintest = PlotElm(20, 2, 90)
# faintest.circleplot('STAR')
# faintest.circleplot('QSO')
# faintest.circleplot('GALAXY')
# faintest.combined()

#faintest.colorplot(['QSO', 'BAL'], 'W2', 'W1', save=False)

#faintest.meanplot(['QSO', 'BAL'], 'z', save=True)
#faintest.meanplot(['QSO', 'BAL'], 'AV', save=True)
#faintest.comparison('QSO')