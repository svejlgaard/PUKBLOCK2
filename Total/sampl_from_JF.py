import numpy as np
import matplotlib.pyplot as plt
import os, contextlib, sys

from astropy.io import fits

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

RA = data['RAdeg_s_gs_gsu']
Dec = data['DEdeg_s_gs_gsu']

pmra = data['pmra_g_gs_gsu']
pmdec = data['pmdec_g_gs_gsu']

pm = np.sqrt(pmra**2 + pmdec**2)

filt = np.nonzero( (errg < 0.5) & (errr < 0.1) & (errJ < 0.1) & (errK < 0.1) )

print(len(RA))

plt.figure()
plt.plot(J[filt]-K[filt],g[filt]-z[filt],color='tab:red',linestyle='', marker='.', markersize=0.3)
plt.xlim(-0.5,4)
plt.ylim(-0.5,7)
plt.xlabel('J-K')
plt.ylabel('g-z')
plt.show()

plt.figure()
plt.plot(J[filt]-K[filt],W1[filt]-W2[filt],color='tab:red',linestyle='', marker='.', markersize=0.1)
plt.xlim(-0.5,2.5)
plt.ylim(-1.5,2)
plt.xlabel('J-K')
plt.ylabel('W1-W2')
filt2 = np.nonzero( (errg < 0.2) & (errr < 0.1) & (errJ < 0.1) & (errK < 0.1) & (W1-W2 < 0.3) & (W1-W2 > 0.1) & (J-K > 1.1) & (J-K < 1.3))
plt.plot(J[filt2]-K[filt2],W1[filt2]-W2[filt2],color='tab:blue',linestyle='', marker='.', markersize=0.1)

plt.show()

#plt.figure()
#plt.plot(RA,Dec,color='tab:red',linestyle='', marker='.')
#plt.show()
print(len(RA))
print(len(RA[filt]))

print(RA[filt2],Dec[filt2])
