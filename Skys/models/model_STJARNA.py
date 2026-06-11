import pandas as pd
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


#############################################
# Nominal model from Stjornumal 
#############################################

def draw_model_param_stjarna(priors_dict):
    """
    Priors are updated with values from the yaml, but are directly assigned here. 
    """
    #First the c block
    mu0, sigma0 = priors_dict['SIM_c']
    mu_c = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_c = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    mu0, sigma0 = priors_dict['SIM_x1']
    mu_x1 = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_x1 = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    mu0, sigma0 = priors_dict['SIM_beta']
    beta = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    mu0, sigma0 = priors_dict['SIM_RV']
    mu_rv = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_rv = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    tau0 = priors_dict['SIM_EBV'][0]
    tau = np.random.uniform(low=tau0[0], high=tau0[1], size=1)
    scatter0 = priors_dict['SCATTER'][0]
    scatter = np.random.uniform(low=scatter0[0], high=scatter0[1], size=1)
    return (mu_c, sig_c, mu_x1, sig_x1, beta, mu_rv, sig_rv, tau, scatter)


def initialise_model_stjarna(sncosmo_model, theta):

    (mu_c, sig_c, mu_x1, sig_x1, beta, mu_rv, sig_rv, tau, scatter) = theta
    
    SNeIa = dict( redshift = {"kwargs": {"zmax":0.2}, "as":"z"},
                           
                        x1 = {"func": scipy.stats.norm.rvs, 
                            "kwargs": {"xx":"-4:4:0.005", 
                                       "loc":mu_x1, 
                                       "scale":sig_x1}}, 
                       
                       c = {"func": scipy.stats.norm.rvs,
                           "kwargs": {"xx":"-0.3:1:0.001", 
                                      "loc":mu_c, 
                                      "scale":sig_c}},
    
                       t0 = {"func": np.random.uniform, 
                             "kwargs": {"low":56_000, "high":56_200} },
                           
                       magabs = {"func": skysurvey.target.snia.SNeIaMagnitude.tripp1998,
                                 "kwargs": {"x1": "@x1", "c": "@c",
                                            "mabs":-19.3,
                                            "sigmaint":scatter, 
                                            "alpha":-0.14, 
                                            "beta":beta}},
    
                        magobs = {"func": "magabs_to_magobs", # str-> method of the class
                                 "kwargs": {"z":"@z", "magabs":"@magabs"}},
                           
                       radec = {"func": skysurvey.tools.utils.random_radec,
                                "kwargs": {},
                                "as": ["ra","dec"]})
    
    #mw_dust = {'effect': sncosmo.models.F99Dust(r_v=3.1),
    #                 'name': 'mw',
    #                 'frame': 'obs',
    #                 'model': {'mwebv': {'func': skysurvey.effects.milkyway.get_mwebv,
    #                     'kwargs': {'ra': '@ra', 'dec': '@dec', 'which':'sfd'}}}}
    
    host_dust = {'effect': sncosmo.models.F99Dust(),
                     'name': 'host',
                     'frame': 'rest',
                     'model': {'hostebv': {"func": SKYexponential, "kwargs": {"xx":"0:2:0.001", "tau":tau}},
                              'hostr_v': {"func": scipy.stats.norm.rvs, "kwargs": {"loc":mu_rv, "scale":sig_rv}}}}
    snia = skysurvey.SNeIa()
    snia.set_model(SNeIa)
    snia.set_rate(23500.0*2) # Fudging the rate so that we get enough SNe
    snia.set_template(sncosmo_model)
    #snia.set_cosmology(cosmo) Will need to come back for this later ... 
    #snia.add_effect(mw_dust)
    snia.add_effect(host_dust)

    return snia 

#############################################
# Next model and so on ... 
#############################################