import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
import os, contextlib, sys

# Sets the directory to the current directory
os.chdir(sys.path[0])


# Read Gaia table
hdul = fits.open('GaiaSDSSUKIDSSAllWISE.fits')
data = hdul[1].data
hdr = hdul[1].header

u = data['umag_s_gs_gsu']
erru = data['e_umag_s_gs_gsu']
g = data['gmag_s_gs_gsu']
errg = data['e_gmag_s_gs_gsu']
r = data['rmag_s_gs_gsu']
errr = data['e_rmag_s_gs_gsu']
i = data['imag_s_gs_gsu']
erri = data['e_imag_s_gs_gsu']
z = data['zmag_s_gs_gsu']
errz = data['e_zmag_s_gs_gsu']

Y = data['Ymag_u_gsu']
errY = data['e_Ymag_u_gsu']
J = data['Jmag_u_gsu']
errJ = data['e_Jmag_u_gsu']
H = data['Hmag_u_gsu']
errH = data['e_Hmag_u_gsu']
K = data['Kmag_u_gsu']
errK = data['e_Kmag_u_gsu']

W1 = data['W1mag_w']
errW1 = data['e_W1mag_w']
W2 = data['W2mag_w']
errW2 = data['e_W2mag_w']
W3 = data['W3mag_w']
errW3 = data['e_W3mag_w']
W4 = data['W4mag_w']
errW4 = data['e_W4mag_w']

RA = data['RAdeg_s_gs_gsu']
Dec = data['DEdeg_s_gs_gsu']

pmra = data['pmra_g_gs_gsu']
pmdec = data['pmdec_g_gs_gsu']
pm = np.sqrt(pmra**2 + pmdec**2)

#Quasar catalog
#First read the master list
#Load spectrum
master = open('MasterCatalogue.dat','r')
header = master.readline()
print(header)
masterdata = [] #List of dictionaries
for line in master:
    line = line.strip()
#    print(repr(line))
    columns = line.split()
    source = {}
    source['name'] = columns[0]
    source['ra'] = columns[1]
    source['dec'] = columns[2]
    source['qso'] = float(columns[3])
    source['bal'] = float(columns[4])
    source['redshift'] = float(columns[5])
    source['u'] = float(columns[6])
    source['uerr'] = float(columns[7])
    source['g'] = float(columns[8])
    source['gerr'] = float(columns[9])
    source['r'] = float(columns[10])
    source['rerr'] = float(columns[11])
    source['i'] = float(columns[12])
    source['ierr'] = float(columns[13])
    source['z'] = float(columns[14])
    source['zerr'] = float(columns[15])
    source['Y'] = float(columns[16])
    source['Yerr'] = float(columns[17])
    source['J'] = float(columns[18])
    source['Jerr'] = float(columns[19])
    source['H'] = float(columns[20])
    source['Herr'] = float(columns[21])
    source['K'] = float(columns[22])
    source['Kerr'] = float(columns[23])
    source['W1'] = float(columns[24])
    source['W1err'] = float(columns[25])
    source['W2'] = float(columns[26])
    source['W2err'] = float(columns[27])
    source['W3'] = float(columns[28])
    source['W3err'] = float(columns[29])
    source['W4'] = float(columns[30])
    source['W4err'] = float(columns[31])
    masterdata.append(source)
#Convert to Pandas data frame
masterdata = pd.DataFrame(masterdata)
filt = np.nonzero( (errg < 0.5) & (errr < 0.1) & (errJ < 0.1) & (errK < 0.1) )

plt.figure()
plt.plot(J[filt]-K[filt],g[filt]-z[filt],color='tab:red',linestyle='', marker='.', markersize=0.5)
plt.xlim(-0.5,4)
plt.ylim(-0.5,7)
plt.xlabel('J-K')
plt.ylabel('g-z')

plt.plot(masterdata.J[masterdata.qso==1]-masterdata.K[masterdata.qso==1],masterdata.g[masterdata.qso==1]-masterdata.z[masterdata.qso==1],color='tab:blue',linestyle='', marker='.', markersize=1.0)

plt.show()

plt.figure()
plt.plot(J[filt]-K[filt],W1[filt]-W2[filt],color='tab:red',linestyle='', marker='.', markersize=0.1)
plt.xlim(-0.5,2.5)
plt.ylim(-1.5,5)
plt.xlabel('J-K')
plt.ylabel('W1-W2')
#filt2 = np.nonzero( (errg < 0.2) & (errr < 0.1) & (errJ < 0.1) & (errK < 0.1) & (W1-W2 < 0.3) & (W1-W2 > 0.1) & (J-K > 1.1) & (J-K < 1.3))
#plt.plot(J[filt2]-K[filt2],W1[filt2]-W2[filt2],color='tab:blue',linestyle='', marker='.', markersize=0.1)


plt.plot(masterdata.J[masterdata.qso==1]-masterdata.K[masterdata.qso==1],masterdata.W1[masterdata.qso==1]-masterdata.W2[masterdata.qso==1],color='tab:blue',linestyle='', marker='.', markersize=1.0)

plt.show()
