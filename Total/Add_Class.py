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
import time

# Sets the directory to the current directory
os.chdir(sys.path[0])

time_signature = datetime.now().strftime("%m%d-%H%M")

random_state = 27

plt.style.use('seaborn')

# Load the fits-file
fitscat = Table.read('GaiaSDSSUKIDSSAllWISE.fits','fits').to_pandas()
rawname = np.array(fitscat['Name_u_gsu'].to_numpy(), dtype=str)


objid = fitscat['ObjID_s_gs_gsu'].to_numpy()
ra = fitscat['Ra_g_gs_gsu'].to_numpy()
dec = fitscat['Dec_g_gs_gsu'].to_numpy()

# Get the object spectral IDs if it is there
specid = fitscat['SpObjID_s_gs_gsu'].to_numpy()
urls = np.array(specid, dtype=object)
for n,s in enumerate(specid):
    if s != 0:
        # Save as url to scrape the subcategories
        urls[n] = f'http://skyserver.sdss.org/dr12/en/get/SpecById.ashx?id={s}'
    else:
        urls[n] = ''


# Load the ascii file
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

# Because there's no spectral IDs, use the coordinates to search on SDSS
ra = masterframe['RA'].to_numpy()
dec = masterframe['Dec'].to_numpy()
extra_specid = list()

for r, d in tqdm(zip(ra, dec)):
    co = coords.SkyCoord(r,d, unit=(u.hourangle, u.deg))
    # Perform the search
    result = SDSS.query_region(co, data_release=12, spectro=True)
    if result == None:
        extra_specid.append('')
    else:
        # Save the id, if it is there
        extra_specid.append(result['specobjid'][-1])

# Similar to what was done for the fits file objects
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

        # If the object has a spectral ID, scrape SDSS to find the plot of the spectrum
        if wanted_spec != '':
            time.sleep(0.1)
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
            # Find the subcategory on the spectrum figure
            text = image_to_string(im, lang="eng").lower()
            text = text[text.find("class=")+6:].split("\n")[0]
            if '0' in text:
                text = text.replace('0','o')
            specid_list[i] = text
        else:
            specid_list[i] = ''
        try:
            class_list[i] = wanted_class[0]
        except:
            class_list[i] = wanted_class
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
    # Save all the data found in SDSS and MasterList
    alldata['Label'] = class_list
    alldata['ID'] = id_list
    alldata['z'] = z_list
    alldata['AV'] = av_list
    alldata['Subclass'] = specid_list

    example = example.split('.')[0]
    alldata.to_csv(f'{example}_labels.csv')

# Example use, folder can't be found on github

list19 = list(glob('G19/final/*.csv'))

for cl in list19:
    print(cl)
    addclass(cl)
