import numpy as np
import pandas as pd 
from astropy.cosmology import Planck18
import scipy
import matplotlib.pyplot as plt
import skysurvey
import astropy
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import yaml
import sncosmo
from scipy.special import expit
import multiprocessing as mp
from Functions import SKYexponential

from skysurvey_models import draw_model_param_stjarna, initialise_model_stjarna

def initialise_ztf():
    """
    Initialisation of ZTF-specific components in skysurvey.
    """
    sncosmo_model = sncosmo.Model(source=sncosmo.get_source('salt3'))

    logs = pd.read_parquet("logs/ztf_logs_coadded.parquet")
    logs.rename(columns={'mjd_round': 'mjd'}, inplace=True)
    logs = logs[logs['mjd'] < 59304]
    ztf = skysurvey.ZTF.from_pointings(data=logs)
    coefs = {"ztfg": 1.23, 
            "ztfr":1.17, 
            "ztfi": 1.2}

    ztf.data["skynoise_orig"] = ztf.data["skynoise"].copy()
    s = ztf.data.groupby("band", group_keys=True)["skynoise_orig"]
    skynoise = pd.concat([s.get_group(f_)*coefs[f_] for f_ in coefs.keys()])
    ztf.data["skynoise"] = skynoise

    return ztf, sncosmo_model


def run_ztf(snia, ztf, i,):
    snia_data = snia.draw(zmin=0.01, 
                          zmax=0.3, 
                          tstart=min(ztf.data['mjd'])-20, 
                          tstop=max(ztf.data['mjd'])+20, 
                          set_amplitude=False,
                          radec={"dec_range":[-28, 90]} 
                         )
    
    snia_data['selected'] = np.random.binomial(1, 1-expit((snia_data['magobs']-18.4)*4.5)) 
    snia_data = snia_data[snia_data['selected'] == 1]
    snia_data.drop(columns={'selected'}, inplace=True)
    snia_data.reset_index(drop=True, inplace=True)      
    snia_data['sn'] = ['ZTF_'+str(k) for k in list(snia_data.index)] # Naming the SNe
    snia.set_data(snia_data)
    
    dset = skysurvey.DataSet.from_targets_and_survey(snia, ztf, incl_error=True, phase_range=[-20, 50])
    index = list(dset.data.index.get_level_values(0))
    dset_data = dset.data.assign(sn=['ZTF_'+str(ind) for ind in index], magsys='ab')
    
    snid = np.unique(dset_data['sn'])
    snia_data = snia_data[snia_data['sn'].isin(snid)]
    print("Brodie note that we're temporarily saving to a random place! ")
    snia_data.to_parquet('simulations/snias_'+str(i)+'.parquet') #truth values
    dset_data.to_parquet('simulations/dset_'+str(i)+'.parquet') #light curves 

    # Here run the SALT fits and save them
    #Future carveout to fit stuff 

    return np.array([i, mu_c, sig_c, mu_x1, sig_x1, beta, mu_rv, sig_rv, tau, scatter])


def run_ztf_worker(i):
    ztf, sncosmo_model = initialise_ztf()
    snia = initialise_model_stjarna(sncosmo_model)
    return run_ztf(snia, ztf, i)


if __name__ == '__main__':


    with mp.Pool(50) as pool:
        results = pool.map(run_ztf_worker, range(10))