import pandas as pd
import numpy as np
import pandas as pd 
from astropy.cosmology import Planck18
import scipy
import matplotlib.pyplot as plt
import skysurvey
import healpy as hp
import astropy.units as u
import yaml
import sncosmo
import multiprocessing as mp
from Functions import SKYexponential, SKYexponential_split, SKYtruncnorm_split
from scipy.stats import skewnorm

#############################################
# Nominal model from Stjornumal 
#############################################

def draw_model_param_stjarna(priors_dict):
    """
    Priors are updated with values from the yaml, but are directly assigned here. 
    """
    priors_sum = sum(len(v) for v in priors_dict.values())
    #Temporarily disabled assertion for testing. 
    assert priors_sum == 15, f"You passed {priors_sum} definitions for the priors, but I was expecting 15. Please look at the yml and your skysurvey model."
    mu0, sigma0 = priors_dict['SIM_c']
    mu_c = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_c = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    mu01, sigma01, mu02, sigma02, ratio = priors_dict['SIM_x1']
    mu1_x1 = np.random.uniform(low=mu01[0], high=mu01[1], size=1)
    sig1_x1 = np.random.uniform(low=sigma01[0], high=sigma01[1], size=1)
    mu2_x1 = np.random.uniform(low=mu02[0], high=mu02[1], size=1)
    sig2_x1 = np.random.uniform(low=sigma02[0], high=sigma02[1], size=1)
    ratio_x1 = np.random.uniform(low=ratio[0], high=ratio[1], size=1)
    mu0, sigma0 = priors_dict['SIM_beta']
    beta = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    mu0, sigma0 = priors_dict['SIM_RV']
    mu_rv_LM = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_rv_LM = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    mu_rv_HM = np.random.uniform(low=mu0[0], high=mu0[1], size=1)
    sig_rv_HM = np.random.uniform(low=sigma0[0], high=sigma0[1], size=1)
    tau0 = priors_dict['SIM_EBV'][0]
    tau_LM = np.random.uniform(low=tau0[0], high=tau0[1], size=1)
    tau_HM = np.random.uniform(low=tau0[0], high=tau0[1], size=1)
    scatter0 = priors_dict['SCATTER'][0]
    scatter = np.random.uniform(low=scatter0[0], high=scatter0[1], size=1)
    return (mu_c, sig_c, mu1_x1, sig1_x1, mu2_x1, sig2_x1, ratio_x1, beta, mu_rv_LM, sig_rv_LM, mu_rv_HM, sig_rv_HM, tau_LM, tau_HM, scatter)


def initialise_model_stjarna(theta):

    (mu_c, sig_c, mu1_x1, sig1_x1, mu2_x1, sig2_x1, ratio_x1, beta, mu_rv_LM, sig_rv_LM, mu_rv_HM, sig_rv_HM, tau_LM, tau_HM, scatter) = theta

    SNeIa = dict( redshift = {"kwargs": {"zmax":0.2}, "as":"z"},
                       
                    c = {"func": scipy.stats.norm.rvs,
                           "kwargs": {"xx":"-0.3:1:0.001", 
                                      "loc":mu_c, 
                                      "scale":sig_c}},
    
                    t0 = {"func": np.random.uniform, 
                             "kwargs": {"low":56_000, "high":56_200} },
                           
                    magobs = {"func": "magabs_to_magobs", # str-> method of the class
                                 "kwargs": {"z":"@z", "magabs":"@magabs"}},
                           
                    radec = {"func": skysurvey.tools.utils.random_radec,
                                "kwargs": {},
                                "as": ["ra","dec"]},
                
                    mass={"func": skewnorm.rvs,
                            "kwargs":{"a":-5.2, "loc":10.894, "scale":1.29}
                          },
             
                    x1 = {"func": mass_to_stretch,
                               "kwargs":{"mass":"@mass", "a":ratio_x1, "mu1":mu1_x1, "sigma1":sig1_x1, "mu2":mu2_x1, "sigma2":sig2_x1}
                          }, 
                   
                    magabs = {"func": skysurvey.target.snia.SNeIaMagnitude.tripp1998,
                                 "kwargs": {"x1": "@x1", "c": "@c",
                                            "mabs":-19.3,
                                            "sigmaint":scatter, 
                                            "alpha":-0.14, 
                                            "beta":beta}}
                    )
    
    host_dust = {'effect': sncosmo.models.F99Dust(),
                     'name': 'host',
                     'frame': 'rest',
                     'model': {'hostebv': {"func": SKYexponential_split, 
                                           "kwargs": {"xx":"0:2:0.001", "tau_HM":tau_HM, "tau_LM":tau_LM, "tracer":"@mass"}},
                              'hostr_v': {"func": SKYtruncnorm_split, 
                                          "kwargs": {"xx":"1:6:0.001", "mu_HM":mu_rv_HM, "sig_HM":sig_rv_HM, "mu_LM":mu_rv_LM, "sig_LM":sig_rv_LM,
                                                     "tracer":"@mass"}}}}
    snia = skysurvey.SNeIa()
    snia.set_model(SNeIa)
    def rate_perley(z, fudge_factor=1.5): # Fudging the rate so that we get enough SNe
        return fudge_factor*23500/(1+z)
    snia.set_rate(rate_perley)
    snia.set_template(sncosmo.Model(source=sncosmo.get_source('salt3')))
    #snia.set_cosmology(cosmo) Will need to come back for this later ... 
    snia.add_effect(skysurvey.effects.mw_extinction)
    snia.add_effect(host_dust)

    return snia 

#############################################
# Next model and so on ... 
#############################################