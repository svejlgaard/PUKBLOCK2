import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
import os, contextlib, sys
from datetime import datetime
from glob import glob
from astropy.io import fits
from astropy.table import Table
import csv
from tqdm import tqdm
from PyAstronomy import pyasl
from astropy import coordinates as coords
from astroquery.sdss import SDSS
import io
import requests
from PIL import Image
from pytesseract import image_to_string
import tempfile
import astropy.units as u

# # Coordinates of HD 1 from SIMBAD
# hd1 = "00 05 08.83239 +67 50 24.0135"

# print("Coordinates of HD 1 (SIMBAD): ", hd1)

# # Obtain decimal representation
# ra, dec = pyasl.coordsSexaToDeg(hd1)
# print("Coordinates of HD 1 [deg]: %010.6f  %+09.6f" % (ra, dec))

# # Convert back into sexagesimal representation
# sexa = pyasl.coordsDegToSexa(ra, dec)
# print("Coordinates of HD 1 [sexa]: ", sexa)


# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

# ObjID_s_gs_gsu

fitscat = Table.read('GaiaSDSSUKIDSSAllWISE.fits','fits').to_pandas()
rawname = np.array(fitscat['Name_u_gsu'].to_numpy(), dtype=str)


objid = fitscat['ObjID_s_gs_gsu'].to_numpy()
ra = fitscat['Ra_g_gs_gsu'].to_numpy()
dec = fitscat['Dec_g_gs_gsu'].to_numpy()


specid = fitscat['SpObjID_s_gs_gsu'].to_numpy()
urls = np.array(specid, dtype=object)
for n,s in enumerate(specid):
    if s != 0:
        urls[n] = f'http://skyserver.sdss.org/dr12/en/get/SpecById.ashx?id={s}'
    else:
        urls[n] = ''


with open("MasterList", "r") as masterfile:
    # Exclude comment col
    additional = {
        key : [] for key in   masterfile.readlines()[0].split()[:-1]
    }
    masterfile.seek(0)
    for row in masterfile.readlines()[1:]:
        row = row.split()
        if not row[0].strip().startswith("#" ):
            additional["NTT_study"].append(row[0])
            additional["RA"].append(":".join(row[1:4]))
            additional["Dec"].append(":".join(row[4:7]))
            additional["QSO"].append(row[7]) 
            additional["BAL"].append(row[8])
            additional["z"].append(row[9])
            additional["AV"].append(row[10])

masterframe = pd.DataFrame.from_dict(additional)
masternames = masterframe['NTT_study'].to_numpy(dtype=str)
print(masterframe)

ra = masterframe['RA'].to_numpy()
dec = masterframe['Dec'].to_numpy()

print(ra[0])

extra_specid = list()

for r, d in tqdm(zip(ra, dec)):
    co = coords.SkyCoord(r,d, unit=(u.hourangle, u.deg))
    result = SDSS.query_region(co, data_release=12, spectro=True)
    if result == None:
        extra_specid.append('')
    else:
        extra_specid.append(result['specobjid'][-1])

masterurls = np.array(extra_specid, dtype=object)
for n,s in enumerate(extra_specid):
    if s != '':
        masterurls[n] = f'http://skyserver.sdss.org/dr12/en/get/SpecById.ashx?id={s}'
    else:
        masterurls[n] = ''


def addclass(clname):

    example = clname

    allclass = pd.read_csv('NameFrame.txt', sep=' ')

    allnames = allclass['Name'].to_numpy(dtype=str)
    alllabels = allclass['Label'].to_numpy(dtype=str)

    allzs = masterframe['z'].to_numpy()
    allavs = masterframe['AV'].to_numpy()
    allspecids = np.ones_like(extra_specid)

    alldata = pd.read_csv(example)

    class_list = np.ones_like(alldata['Name'].to_numpy())

    id_list = np.ones_like(alldata['Name'].to_numpy())

    z_list = np.ones_like(alldata['Name'].to_numpy())

    av_list = np.ones_like(alldata['Name'].to_numpy())

    specid_list = np.ones_like(alldata['Name'].to_numpy())

    for i, n in tqdm(enumerate(alldata['Name'].to_numpy(dtype=str))):
        wanted_class = alllabels[np.where(allnames == n)]
        wanted_id = objid[np.where(rawname == n)]
        wanted_z = allzs[np.where(masternames == n)]
        wanted_av = allavs[np.where(masternames == n)]
        if n in rawname:
            wanted_spec = urls[np.where(rawname == n)][0]
        if n in masternames:
            wanted_spec = masterurls[np.where(masternames == n)][0]
            #print(wanted_spec)

        if wanted_spec != '':
            #wanted_spec = wanted_spec[0]
            buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
            try:
                r = requests.get(wanted_spec, stream=True)
            except Exception as e:
                print(e)
                print(f'Object name {n} with id {wanted_id}')
                continue
            if r.status_code == 200:
                downloaded = 0
                filesize = int(r.headers['content-length'])
                for chunk in r.iter_content():
                    downloaded += len(chunk)
                    buffer.write(chunk)
                buffer.seek(0)
                im = Image.open(io.BytesIO(buffer.read()))
                im = np.asarray(im)
            buffer.close()

            text = image_to_string(im, lang="eng").lower()
            text = text[text.find("class=")+6:].split("\n")[0]
            if '0' in text:
                text = text.replace('0','o')
            specid_list[i] = text
        else:
            specid_list[i] = ''
        #wanted_class = allclass['Label'].loc[allclass['Name'] == n]
        class_list[i] = wanted_class[0]
        
        try:
            id_list[i] = wanted_id[0]
        except:
            id_list[i] = wanted_id
        try:
            z_list[i] = wanted_z[0]
        except:
            z_list[i] = wanted_z
        try:
            av_list[i] = wanted_av[0]
        except:
            av_list[i] = wanted_av
    alldata['Label'] = class_list
    alldata['ID'] = id_list
    alldata['z'] = z_list
    alldata['AV'] = av_list
    alldata['Subclass'] = specid_list

    example = example.split('.')[0]

    alldata.to_csv(f'{example}_labels.csv')

for cl in glob('G18/*.csv'):
    print(cl)
    addclass(cl)

for cl in glob('G20/*.csv'):
    print(cl)
    addclass(cl)

for cl in glob('G19/*p50_G19.csv'):
    print(cl)
    addclass(cl)
