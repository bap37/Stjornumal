import numpy as np
import pandas as pd 
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import skysurvey
import sncosmo
from scipy.special import expit
from pathos.multiprocessing import ProcessingPool as Pool
from pathlib import Path



################################
# The model_chooser
################################

def model_chooser(infos):
    """
    The beating heart of the skysurvey integration. After building your desired model in Skys/models/ (see model_STJARNA.py for an example), add it here.
    Should 

    """
    if infos['Skysurvey'] == 'Stjarna':
        from Skys.models.model_STJARNA import initialise_model_stjarna, draw_model_param_stjarna
    #Import other models here
    else: 
        print(f"I did not recognise {infos['Skysurvey']}; it is not implemented ")
        quit()

    return initialise_model_stjarna, draw_model_param_stjarna


################################
# Skysurvey Simulation Code
################################


def simulate_model_lightcurves_skysurvey(infos, simulator, theta_generator, model_initialiser, sncosmo_model, survey_information, device="cpu"):

    from tqdm import tqdm

    n_sim = infos['sim_parameters']['n_sim']
    n_batch = infos['sim_parameters']['n_batch']
    sims_savename = infos['sim_parameters']['simname']

    #come back to this... 
    print(sims_savename.split("/")[0])
    outdir = Path(sims_savename.split("/")[0]+"/TMP")
    outdir.mkdir(exist_ok=True)

    def run_single_sim(i):

        theta = theta_generator(infos["Priors"])
        snia = model_initialiser(sncosmo_model, theta)

        simulator(
            snia,
            survey_information,
            sim_id=i,
            savename=sims_savename
        )

        return i

    with Pool(processes=n_batch) as pool:

        results = list(
            tqdm(
                pool.uimap(run_single_sim, range(n_sim)),
                total=n_sim,
            )
        )
                # - END UPDATES

    return 

####################################
# Initialisation functions for ZTF in skysurvey
####################################

def initialise_ztf():
    """
    Initialisation of ZTF-specific components in skysurvey.
    """
    sncosmo_model = sncosmo.Model(source=sncosmo.get_source('salt3'))

    logs = pd.read_parquet("Skys/logs/ztf_logs_coadded.parquet")
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


#Will eventually need to update this to be more general and not just ztf-specific... 
def run_ztf(snia, ztf, sim_id=None, savename=None):
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

    #    simname: 'simulations/sims.v5.NOM.h5'
    snia_data.to_parquet(
         f"{savename.split('/')[0]}/TMP/{sim_id:06d}_truth.parquet"
    )

    dset_data.to_parquet(
        f"{savename.split('/')[0]}/TMP/{sim_id:06d}_lightcurves.parquet"
    )

    # Here run the SALT fits and save them
    #Future carveout to fit stuff 

    return snia_data, dset_data
