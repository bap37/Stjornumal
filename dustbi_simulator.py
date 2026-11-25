from dataclasses import dataclass
import torch
from torch.distributions import Normal, LogNormal, Exponential, HalfNormal
from sbi.utils import MultipleIndependent
import numpy as np
from sbi.utils import BoxUniform
from Functions import *


#######################
## BEGIN SIMULATOR
#######################


#####################
### BEGIN SUPPORT FUNCTIONS
###########################

from scipy.interpolate import BSpline
import numpy as np
import torch


def build_bspline_basis(x, x_min, x_max, n_coeffs, degree=3):
    """
    Build a cubic B-spline basis with exactly n_coeffs basis functions.

    Parameters
    ----------
    x : torch.Tensor (N,)
        Evaluation locations.
    x_min, x_max : float
        Domain of the spline.
    n_coeffs : int
        Number of spline coefficients (basis functions).

    Returns
    -------
    basis : torch.Tensor (N, n_coeffs)
    """

    x = x.detach().cpu().numpy()

    # Number of interior knots required
    n_internal = max(n_coeffs - degree - 1, 0)

    if n_internal > 0:
        internal = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    else:
        internal = np.array([])

    # Open-uniform knot vector
    t = np.concatenate([
        np.repeat(x_min, degree + 1),
        internal,
        np.repeat(x_max, degree + 1),
    ])

    basis = np.empty((len(x), n_coeffs))

    for i in range(n_coeffs):
        coeffs = np.zeros(n_coeffs)
        coeffs[i] = 1.0

        spline = BSpline(t, coeffs, degree, extrapolate=False)
        basis[:, i] = spline(x)

    basis = np.nan_to_num(basis)

    return torch.tensor(basis, dtype=torch.float32)

def preprocess_input_distribution(df, cols):
    return {
        col: torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        for col in cols
    }



def check_special_inputs(infos):
    try: 
        infos['Splits']
    except KeyError:
        infos['Splits'] = {}
        print("Temporarily hacking splits to be an empty dict")
    
    try:
        infos['selection_parameters']
    except KeyError:
        infos['selection_parameters'] = {}

    try:
        infos['Skysuryey']
    except KeyError:
        infos['Skysurvey'] = False

    return infos

def preprocess_data(parameters_to_condition_on, dfdata, ):
    
    output_distribution = preprocess_input_distribution(dfdata, parameters_to_condition_on)

    x = torch.stack(
        [output_distribution[p] for p in parameters_to_condition_on],
        dim=-1
    )
    
    return x


def load_kestrel(filename):
    
    import ast
    import yaml
    
    FUNCTION_REGISTRY = {
        "DistGaussian": DistGaussian,
        "DistExponential": DistExponential,
        "DistLogistic": DistLogistic,
        "DistDoubleGaussian": DistDoubleGaussian,
        "DistGamma": DistGamma,
        "DistIdentity": DistIdentity,
        "host_step":host_step,
    }


    with open(filename, 'r') as file:
        raw_yaml = yaml.safe_load(file)

    raw_yaml['param_names'] = raw_yaml['param_names'].split(" ")
    raw_yaml['nuisance_parameters'] = raw_yaml['nuisance_parameters'].split(" ")
        
    raw_yaml['Data_File'] = raw_yaml['Data_File'].split(" ")
    raw_yaml['Simbank_File'] = raw_yaml['Simbank_File'].split(" ")



    _priors = raw_yaml['Priors']
    for entry in _priors:
        for n in range(len(_priors[entry])):
            raw_yaml['Priors'][entry][n] = ast.literal_eval(raw_yaml['Priors'][entry][n])
            
    raw_yaml['Functions'] = {
        name: FUNCTION_REGISTRY[func_name]
        for name, func_name in raw_yaml["Functions"].items()
        }

    try:
        raw_yaml['Models_Comparison'] = raw_yaml['Models_Comparison'].split(" ")
    except KeyError:
        pass

    if 'Population_B' in raw_yaml:
        pop_b = raw_yaml['Population_B']
        for entry in pop_b['Priors']:
            for n in range(len(pop_b['Priors'][entry])):
                pop_b['Priors'][entry][n] = ast.literal_eval(pop_b['Priors'][entry][n])
        pop_b['mixing_prior'] = ast.literal_eval(pop_b['mixing_prior'])

    return raw_yaml


def add_distance(df_tensor):
    
    x1_obs = df_tensor['x1'] ; c_obs = df_tensor['c'] ; mB_obs = df_tensor['mB']
    
    beta = 3.1 ; alpha = 0.16 ; M0 = -19.3
    
    correction = alpha * x1_obs - beta * c_obs + M0 + mB_obs

    #print("Setting the mean MURES values to around 0.")
    MURES =  correction - df_tensor['MU'] #- torch.mean(correction) + torch.mean(df_tensor['MU'])
    
    return  MURES


def load_data(simfilename, datfilename):

    from astropy.cosmology import Planck18
    import numpy as np
    import pandas as pd
    df = pd.read_csv(simfilename, comment="#", sep='\s+')

    df['SIM_EBV'] = df.SIM_AV/df.SIM_RV
    df['MU'] = Planck18.distmod(df.zHD.values).value


    
    dfdata = pd.read_csv(datfilename, 
                             comment="#", sep=r'\s+')
    dfdata['MU'] = Planck18.distmod(dfdata.zHD.values).value

    print("Only loading DES Data")
    #dfdata = dfdata.loc[dfdata.IDSURVEY == 10]
    #dfdata.loc[dfdata.PROB_SNNV19 < -1., "PROB_SNNV19"] = 1
    try:
        dfdata = dfdata.loc[dfdata.PROB_SNNV19 >= 0.5]
    except AttributeError:
        print("This file doesn't have 'PROB_SNNV19' as a valid key.")

    print("Ensuring only valid log masses.")
    dfdata = dfdata.loc[dfdata.HOST_LOGMASS > 0]
    df = df.loc[df.HOST_LOGMASS > 0 ]

    return df, dfdata

#######################
## Priors and such ! 
#######################


def prior_generator(infos):

    priors_dict = {}
    
    for name in infos['param_names']:
        
        priors_dict[name] = build_distribution_priors(name, infos, device='cpu')

    for name in infos['nuisance_parameters']:
        try:
            func_name = infos['Functions'][name].__name__
        except KeyError:
            func_name = name 
        if name == "2POPS":
            priors_dict['a'] = build_special_priors(name, infos, device='cpu')
        else:
            print(name)
            priors_dict[name] = build_special_priors(name, infos, device='cpu')

    #Then slot in any additional correlations.
    for name in infos['Correlations'].keys():
        priors_dict = build_correlation_priors(name, infos, priors_dict, device='cpu')

    return priors_dict

def build_distribution_priors(name, infos , device='cpu'):
    """Build the list of BoxUniform priors for distribution parameters.
    Called one at a time and typically handed to a dictionary for safe keeping. 
    
    """

    priors_dict = infos["Priors"]
        
    func_name = infos['Functions'][name].__name__

    if func_name == "DistGaussian":
        mu0, sigma0 = priors_dict[name] 
        
        mu_prior, sigma_prior = TwoDBoxPrior(mu0, sigma0, device=device)
        return MultipleIndependent([mu_prior, sigma_prior])

    elif "DistExponential" in func_name:
        tau0 = priors_dict[name][0]

        tau_prior = BoxUniform(
            low= torch.tensor([tau0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([tau0[1]], dtype=torch.float32, device=device)
                )

        return tau_prior

    elif func_name == "DistDoubleGaussian":
        mu1, sigma1, mu2, sigma2, a = priors_dict[name]

        mu1_prior = BoxUniform(
            low= torch.tensor([mu1[0]], dtype=torch.float32, device=device),
            high=torch.tensor([mu1[1]], dtype=torch.float32, device=device)
            )

        sigma1_prior = BoxUniform(
            low= torch.tensor([sigma1[0]], dtype=torch.float32, device=device),
            high=torch.tensor([sigma1[1]], dtype=torch.float32, device=device)
            )

        mu2_prior = BoxUniform(
            low= torch.tensor([mu2[0]], dtype=torch.float32, device=device),
            high=torch.tensor([mu2[1]], dtype=torch.float32, device=device)
            )

        sigma2_prior = BoxUniform(
            low= torch.tensor([sigma2[0]], dtype=torch.float32, device=device),
            high=torch.tensor([sigma2[1]], dtype=torch.float32, device=device)
            )

        a_prior = BoxUniform(
            low= torch.tensor([a[0]], dtype=torch.float32, device=device),
            high=torch.tensor([a[1]], dtype=torch.float32, device=device)
            )

        return MultipleIndependent([mu1_prior, sigma1_prior, mu2_prior, sigma2_prior, a_prior])

    elif func_name == "DistLogistic":
        L0, k0, sigma0 = priors_dict[name]
        L_prior = BoxUniform(
            low= torch.tensor([L0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([L0[1]], dtype=torch.float32, device=device)
            )
        k_prior = BoxUniform(
            low= torch.tensor([k0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([k0[1]], dtype=torch.float32, device=device)
            )
        sigma_prior = BoxUniform(
            low= torch.tensor([sigma0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([sigma0[1]], dtype=torch.float32, device=device)
            )
        return MultipleIndependent([L_prior, k_prior, sigma_prior])

    elif func_name == "DistGamma":
        tau0, gamma0 = priors_dict[name]
        tau_prior = BoxUniform(
            low= torch.tensor([tau0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([tau0[1]], dtype=torch.float32, device=device)
            )
        gamma_prior = BoxUniform(
            low= torch.tensor([gamma0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([gamma0[1]], dtype=torch.float32, device=device)
            )
        return MultipleIndependent([tau_prior, gamma_prior])

    elif func_name == "DistIdentity":
        ident0 = priors_dict[name][0]

        ident_prior = BoxUniform(
            low= torch.tensor([ident0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([ident0[1]], dtype=torch.float32, device=device)
                )

        return ident_prior

    else:
        AssertionError(f"You passed me {func_name}; I do not recognise this.")


def build_special_priors(name, infos, device='cpu'):
    """Build the list of priors for STEP, SCATTER, SELECTION, and a special key 
    for the mixing parameter a. """

    priors_dict = infos["Priors"]

    #knot_list=None, selection_parameters=None

    if "gammaDM" in name:
        step0 = priors_dict[name][0]
        step_prior = BoxUniform(
            low= torch.tensor([step0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([step0[1]], dtype=torch.float32, device=device)
            )
        return step_prior

    elif "sigint" in name:
        scatter0 = priors_dict[name][0]
        scatter_prior = BoxUniform(
            low= torch.tensor([scatter0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([scatter0[1]], dtype=torch.float32, device=device)
            )
        return scatter_prior

    elif "SELECTION" in name:
        list_o_priors = []
        for s in selection_parameters:
            for _ in range(len(knot_list[s])):
                selection_prior = BoxUniform(
                    low=torch.tensor([0], dtype=torch.float32, device=device),
                    high=torch.tensor([5], dtype=torch.float32, device=device)
                )
                list_o_priors.append(selection_prior)
        return MultipleIndependent(list_o_priors)

    elif "2POPS" in name:
        a_prior = BoxUniform(
            low= torch.tensor([0], dtype=torch.float32, device=device),
            high=torch.tensor([1], dtype=torch.float32, device=device)
            )
        return a_prior
    
def build_correlation_priors(name, infos, priors, device='cpu'):
    """
    Code to add correlation priors to functions. Explicitly excludes functions that already require a correlation;
    e.g. ["DistLogistic", "DistGamma"] which don't work without that second dimension.

    Input:  parameter name (string)
            infos (nested dictionary)
            priors (nested dictionary)
    """

    priors_dict = infos['Priors']

    if infos['Functions'][name].__name__ not in ["DistLogistic", "DistGamma"]:
        corr0 = priors_dict[name+"_EVOL"][0]

        corr_prior = BoxUniform(
            low= torch.tensor([corr0[0]], dtype=torch.float32, device=device),
            high=torch.tensor([corr0[1]], dtype=torch.float32, device=device))
    
        priors[name] = MultipleIndependent(list(priors[name].dists) + [corr_prior])

    else: #For DistLogistic and DistGamma, we do nothing; correlations are already handled.
        pass
    
        
    return priors

def TwoDBoxPrior(param_1, param_2, device="cpu"):
    """
    General wrapper for any distribution with two parameters.
    """

    p1_prior = BoxUniform(
        low= torch.tensor([param_1[0]], dtype=torch.float32, device=device),
        high=torch.tensor([param_1[1]], dtype=torch.float32, device=device)
        )

    p2_prior = BoxUniform(
        low= torch.tensor([param_2[0]], dtype=torch.float32, device=device),
        high=torch.tensor([param_2[1]], dtype=torch.float32, device=device)
        )

    return p1_prior, p2_prior



##################
## Knots creation
##################

def create_knots(infos):
    import pandas as pd

    parameters_to_condition_on = infos['parameters_to_condition_on']
    datfilename = infos['Data_File'][0]

    if infos.get('selection_parameters'):
    
        selection_parameters = infos['selection_parameters']['parameters']
        n_coeff = infos['selection_parameters']['n_coeff']
        
        dfdata = pd.read_csv(
            datfilename,
            comment="#",
            sep=r'\s+'
        )
    
        # Remove invalid selection values
        for param in selection_parameters:
            dfdata = dfdata.loc[dfdata[param] > 0]
    
        # Build an independent knot grid for each selection parameter
        knot_lists = {}
    
        for param in selection_parameters:
            lower = np.amin(dfdata[param].values)
            upper = np.amax(dfdata[param].values)
    
            knot_lists[param] = np.linspace(
                lower,
                upper,
                n_coeff
            )
    
        # Add selection parameters onto parameters_to_condition_on if needed
        parameters_to_condition_on.extend(
            p for p in selection_parameters
            if p not in parameters_to_condition_on
        )
    
        print(
            f"I am adding {selection_parameters} to parameters_to_condition_on, "
            "if you have not already!"
        )
    
    else:
        knot_lists = None
        infos['selection_parameters'] = {}
        selection_parameters = None

    return knot_lists, parameters_to_condition_on, infos, selection_parameters