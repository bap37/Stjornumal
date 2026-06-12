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

def fit_model_lightcurves_skysurvey(infos, simulator, theta_generator, model_initialiser, sncosmo_model, survey_information, device="cpu"):

    #print(sims_savename.split("/")[0])
    #outdir = Path(sims_savename.split("/")[0]+"/TMP")
    #outdir.rmdir(exist_ok=True)


    return "bloop"

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
    
    # Selection on the light curves
    lcs = dset_data.copy()
    snias = snia_data.copy()

    lcs.reset_index(inplace=True)
    snias.reset_index(inplace=True)
    snias_new = snias.copy()
    
    lcs["ndetection"] = lcs["flux"]/lcs["fluxerr"]
    lcs.set_index('sn', inplace=True)
    snias.set_index('sn', inplace=True)
    data_merged = lcs.merge(snias[["t0", "z", "x1", "c"]], 
                              on="sn")
    data_merged["phase_obs"] = data_merged["mjd"] - data_merged["t0"]
    data_merged["phase"] = data_merged["phase_obs"]/(1+data_merged["z"])
    ndet = data_merged[(data_merged["ndetection"]>=5) &
                       (data_merged["phase"].between(-10, +40))
                      ].groupby(["sn", "band"]).size()
    ndet_pre = data_merged[(data_merged["ndetection"]>=5) &
                           (data_merged["phase"].between(-10, 0))
                      ].groupby(["sn", "band"]).size()
    ndet_post = data_merged[(data_merged["ndetection"]>=5) &
                            (data_merged["phase"].between(0, +40))
                      ].groupby(["sn", "band"]).size()
    nbands = ndet.groupby(level=0).size()
    ndetection = ndet.groupby(level=0).sum()
    ndetection_pre = ndet_pre.groupby(level=0).sum()
    ndetection_post = ndet_post.groupby(level=0).sum()
    flag_used = (nbands>=2) & (ndetection>7) & (ndetection_pre>=2) & (ndetection_post>=2)
    
    targets_to_consider = ndetection[flag_used].index
    lcs.rename(columns={'mjd': 'time', 'magsys': 'zpsys'}, inplace=True)

    # SALT fits
    
    salt_fits = []

    for i in range(len(targets_to_consider)):
        sn_name = targets_to_consider[i]
        lc = lcs[lcs.index == targets_to_consider[i]]
        sn = snias_new[snias_new['sn'] == targets_to_consider[i]]
        df_result = fit_salt_integrated(lc, z_guess=sn['z'].iloc[0], t0_guess=sn['t0'].iloc[0]+norm.rvs(0, 2), mwebv_guess=sn['mwebv'].iloc[0])
        salt_fits.append(np.array(df_result))
    salt_fits = np.array(salt_fits)

    df_salt = pandas.DataFrame(salt_fits, columns=['z', 't0', 'x0', 'x1', 'c', 'mwebv',
                                                 'cov_t0_t0', 'cov_t0_x0', 'cov_t0_x1', 'cov_t0_c',
                                                'cov_x0_t0', 'cov_x0_x0', 'cov_x0_x1', 'cov_x0_c',
                                                'cov_x1_t0', 'cov_x1_x0', 'cov_x1_x1', 'cov_x1_c',
                                                'cov_c_t0', 'cov_c_x0', 'cov_c_x1', 'cov_c_c',
                                                'chisq', 'ndof'])

    df_salt['t0_err'] = np.sqrt(df_salt['cov_t0_t0'])
    df_salt['x0_err'] = np.sqrt(df_salt['cov_x0_x0'])
    df_salt['x1_err'] = np.sqrt(df_salt['cov_x1_x1'])
    df_salt['c_err'] = np.sqrt(df_salt['cov_c_c'])
    df_salt['sn'] = list(targets_to_consider)
    df_salt['fitprob'] = stats.chi2.sf(df_salt['chisq'], df_salt['ndof'])

    # Select on SALT
    
    mask_c = df_salt['c'].between(-0.2, 0.8) & (df_salt['c_err'] < 0.1)
    mask_x1 = df_salt['x1'].between(-3, 3) &  (df_salt['x1_err'] < 1)
    mask_fit = (df_salt['t0_err'] < 2) & (df_salt['fitprob'] > 0.05)
    df_salt_selected = df_salt[mask_c & mask_x1 & mask_fit]

    # Save the SALT fits
    
    df_salt_selected.to_parquet(
        f"{savename.split('/')[0]}/TMP/{sim_id:06d}_salt_fits.parquet"
    )
    #Future carveout to fit stuff 

    return snia_data, dset_data, df_salt_selected


def fit_salt(lc, z_guess, t0_guess, mwebv_guess):
    model = sncosmo.Model(source=sncosmo.get_source('salt3'), effects=[sncosmo.F99Dust(r_v=3.1)], effect_names=['mw'], effect_frames=['obs'])
    model.set(z=z_guess, t0=t0_guess, mwebv=mwebv_guess)
    keymap={}
    lc_dict = {key: lc[keymap.get(key, key)].values for key in ["time", "band", "flux", "fluxerr","zp", "zpsys"]}
    try:
        result, fitted_model = sncosmo.fit_lc(
        lc_dict, model,
        ['t0', 'x0', 'x1', 'c'],  # parameters of model to vary
        bounds={'x1':(-10, 10), 'c':(-1, 3)}, modelcov=True)  # bounds on parameters (if any)
    except RuntimeError :
        return np.ones(24)*np.nan
    if result['success'] == False:
        return np.ones(24)*np.nan
    return np.concatenate([result['parameters'], result['covariance'].flatten(), np.array([result['chisq'], result['ndof']])])
