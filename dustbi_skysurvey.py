import numpy as np
import pandas as pd 
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import skysurvey
import sncosmo
from scipy.special import expit
import multiprocessing as mp

from skysurvey_models import initialise_model_stjarna, draw_model_param_stjarna

def initialise_ztf():
    """
    Initialisation of ZTF-specific components in skysurvey.
    """
    sncosmo_model = sncosmo.Model(source=sncosmo.get_source('salt3'))

    logs = pd.read_parquet("skysurvey/logs/ztf_logs_coadded.parquet")
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

def run_ztf(snia, ztf, theta):
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
    #snia_data.to_parquet('simulations/snias_'+str(i)+'.parquet') #truth values
    #dset_data.to_parquet('simulations/dset_'+str(i)+'.parquet') #light curves 

    # Here run the SALT fits and save them
    #Future carveout to fit stuff 

    return np.array(theta)



#Model chooser should read something from STJARNA.yml and return the appropriate model 
def model_chooser(infos, sncosmo_model):
    if infos['Skysurvey'] == 'Stjarna':
        theta = draw_model_param_stjarna(infos['Priors'])
        quit()
        snia = initialise_model_stjarna(sncosmo_model, theta)
    else: 
        print(f"I did not recognise {infos['Skysurvey']}; it is not implemented ")
        quit()

    return snia,theta

if __name__ == '__main__':

    from dustbi_simulator import load_kestrel
    infos = load_kestrel('config_files/STJARNA.yml')

    ztf, sncosmo_model = initialise_ztf()
    snia,theta = model_chooser(infos, sncosmo_model=sncosmo_model)
    run_ztf(snia, ztf, theta=theta)