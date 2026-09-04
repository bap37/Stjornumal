import pandas as pd
import numpy as np
import math
from dustbi_simulator import *
import argparse



#Define relevant functions
def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
def mu_linear(y, slope):
    return slope * y 

def exponential(x, tau):
    return (1/tau) * np.exp(-x / tau)

def doublegaussian(x, mu1, sigma1, mu2, sigma2, a):

    G1 = np.exp(-0.5 * ((x - mu1)/sigma1)**2) / (sigma1 * math.sqrt(2.0 * math.pi))
    G2 = a*np.exp(-0.5 * ((x - mu2)/sigma2)**2) / (sigma2 * math.sqrt(2.0 * math.pi))

    return G1+G2

xvals = np.linspace(0.4,8,10000)
corvals = np.linspace(8,12,10000)

#Then support functions 
def to_space_separated(data):
    def flatten(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    return " ".join(f"{float(v):.3f}" for v in flatten(data))

def collect_all_params(pdf_entries):
    all_params = []

    for entry in pdf_entries:
        etype = entry["type"]

        # --- 1D ---
        if etype == "1d":
            all_params.extend(entry["param_values"])

        # --- stacked ---
        elif etype == "stacked":
            for vals in entry["param_values"]:
                all_params.extend(vals)

        # --- evolving gaussian ---
        elif etype == "evolving_gaussian":
            # mu parameters
            if "mu_param_values" in entry:
                all_params.extend(entry["mu_param_values"])
            elif "mu" in entry:  # cleaner nested version
                all_params.extend(entry["mu"]["param_values"])

            # sigma
            all_params.append(entry["sigma"])

        else:
            raise ValueError(f"Unknown PDF type: {etype}")

    return all_params

#Need this for DOCNANA
docstring = """DOCUMENTATION:
    PURPOSE:    Population distributions (stretch, intrinsic color, intrinisic beta, RV, E(B-V)) for testing PPC of KESTREL
    INTENT:     Validation
    USAGE_KEY:  GENPDF_FILE
    USAGE_CODE: snlc_sim.exe
    VALIDATE_SCIENCE: Used in validation of KESTREL
    NOTES:
    VERSIONS:
    - DATE: April 26
      AUTHORS: Brodie
DOCUMENTATION_END:

"""


def GENPDF_WRITER(pdf_entries, arrays_list, filename,
                  param_labels=None, param_dict=None, corr_bins=None, 
                  beta=(2,0.5), sigint=0):

    if param_labels is None:
        param_labels = [f"PDF {i}" for i in range(len(pdf_entries))]

    with open(filename, "w") as f:

        f.write(docstring)
        input_labels = " ".join(param_dict.keys())
        input_values = " ".join(
            f"{float(v):.3f}" for v in param_dict.values()
        )

        f.write(f"#INP: {input_labels}\n")
        f.write(f"#  {input_values}\n")
        f.write("\n")



        for entry, arr, label in zip(pdf_entries, arrays_list, param_labels):

            # --- handle arrays ---
            if isinstance(arr, tuple):
                x_array, corr_array = arr
                x_array = np.array(x_array)
                corr_array = np.array(corr_array)
            else:
                x_array = np.array(arr)
                corr_array = None

            etype = entry["type"]

            # =========================================================
            # CASE 1: 1D PDF
            # =========================================================
            if etype == "1d":
                f.write(f"VARNAMES: {label} PROB\n")                
                func = entry["func"]
                params = dict(zip(entry["param_names"], entry["param_values"]))

                probs = func(x_array, **params)
                probs = probs / np.max(probs)

                for xv, pv in zip(x_array, probs):
                    f.write(f"PDF: {xv:.3f} {pv:.3f}\n")

            # =========================================================
            # CASE 2: STACKED
            # =========================================================
            elif etype == "stacked":
                f.write(f"VARNAMES: {label} LOGMASS PROB\n")
                if corr_array is None:
                    raise ValueError("Stacked PDF requires (x_array, corr_array)")
                if corr_bins is None:
                    raise ValueError("corr_bins must be provided")

                funcs = entry["funcs"]
                names = entry["param_names"]
                values = entry["param_values"]

                for yv in corr_array:
                    idx = np.digitize(yv, corr_bins) - 1
                    idx = max(0, min(idx, len(funcs) - 1))

                    func = funcs[idx]
                    params = dict(zip(names[idx], values[idx]))

                    probs = func(x_array, **params)
                    probs = probs / np.max(probs)

                    for xv, pv in zip(x_array, probs):
                        f.write(f"PDF: {xv:.3f} {yv:.2f} {pv:.3f}\n")

            # =========================================================
            # CASE 3: EVOLVING GAUSSIAN
            # =========================================================
            elif etype == "evolving_gaussian":
                f.write(f"VARNAMES: {label} LOGMASS PROB\n")
                if corr_array is None:
                    raise ValueError("Evolving Gaussian requires (x_array, corr_array)")
            
                mu_func = entry["mu_func"]
                sigma = entry["sigma"]
            
                mu_params = dict(zip(entry["mu_param_names"], entry["mu_param_values"]))
            
                for yv in corr_array:
                    mu = mu_func(yv, **mu_params)
                    probs = gaussian(x_array, mu, sigma)
                    probs = probs / np.max(probs)
            
                    for xv, pv in zip(x_array, probs):
                        f.write(f"PDF: {xv:.3f} {yv:.2f} {pv:.3f}\n")
            
            else:
                raise ValueError(f"Unknown PDF type: {etype}")

            f.write("\n")

        # --- footer ---
        f.write("MAG_OFFSET:  -0.12\n")
        f.write("GENPEAK_SALT2ALPHA:  0.15\n")
        f.write(f"GENPEAK_SALT2BETA:  {beta[0]}\n")
        f.write(f"GENSIGMA_SALT2BETA: {beta[1]:.4f} {beta[1]:.4f}\n")
        f.write("GENRANGE_SALT2BETA: 0.4  3.0\n")
        f.write(f"GENMAG_SMEAR: {sigint:.4f}")


#This is where we load in stuff 

def read_params(filename):

    params = {}

    with open(filename, "r") as f:
        for line in f:
            key, value = line.strip().split(",", 1)
            params[key] = float(value)

    return params

def construct_dict_colour(infos, param_dict):
    
    if infos['Functions']['SIM_c'].__name__ == "DistGaussian":
        cdict = {
        "type": "1d",
        "func": gaussian,
        "param_names": ["mu", "sigma"],
        "param_values": [
        param_dict[r'$c_{int}$ $\mu$'],
        param_dict[r'$c_{int}$ $\sigma$']
        ]


        }

    else:
        print("This feature is not enabled yet!")

    return cdict


def construct_dict_RV(infos, param_dict):
     
    if infos['Functions']['SIM_RV'].__name__ == "DistGaussian":
        
        if infos['Splits']['SIM_RV']:

            rvdict = {
                "type": "stacked",
                "funcs": [gaussian, gaussian],
                "param_names": [
                    ["mu", "sigma"],
                    ["mu", "sigma"]
                ],
                "param_values": [
                    [
                        param_dict[r'$R_V$ $\mu$'],
                        param_dict[r'$R_V$ $\sigma$']
                    ],
                    [
                        param_dict[r'$R_V$ Hi $\mu$'],
                        param_dict[r'$R_V$ Hi $\sigma$']
                    ]
                ]
            }

        
        else:
            rvdict = {
                "type": "1d",
                "func": gaussian,
                "param_names": ["mu", "sigma"],
                "param_values": [
                param_dict[r'$R_V$ $\mu$'],
                param_dict[r'$R_V$ $\sigma$']
                ]

                }

    return rvdict

def construct_dict_ebv(infos, param_dict):
    
    if infos['Functions']['SIM_EBV'].__name__ == "DistExponential":
        if infos['Splits']['SIM_EBV']:

            ebv_dict = {
                "type": "stacked",
                "funcs": [exponential, exponential],
                "param_names": [["tau"], ["tau"]],
                "param_values": [[param_dict[r'$EBV$ $\tau$']], [param_dict[r'$EBV$ Hi $\tau$']]]
            }

        
        else:
            ebv_dict = {
                "type": "1d",
                "func": exponential,
                "param_names": ["tau"],
                "param_values": [param_dict[r'$EBV$ $\tau$']]
                    }

    return ebv_dict


def construct_dict_x1(infos, param_dict):

    if infos['Functions']['SIM_x1'].__name__ == "DistGaussian":
        x1dict = {
            "type": "1d",
            "func": gaussian,
            "param_names": ["mu", "sigma"],
            "param_values": [
            param_dict[r'$x_1$ $\mu$'],
            param_dict[r'$x_1$ $\sigma$']
            ]

            }

    elif infos['Functions']['SIM_x1'].__name__ == "DistDoubleGaussian":

        print("Feature not enabled yet ! ")
        quit()

        x1dict = {
            "type": "1d",
            "func": doublegaussian,
            "param_names": ["mu1", "sigma1", "mu2", "sigma2", "a"],
            "param_values": [[param_dict[r'$x_1$ $\mu$']], [param_dict[r'$x_1$ $\sigma$']], 
                            [param_dict[r'$x_1$ $\sigma$']], [param_dict[r'$x_1$ $\sigma$']]
                            [param_dict[r'$x_1$ a']]]
            }


    elif infos['Functions']['SIM_x1'].__name__ == "DistGaussian_EVOL":

        print("Feature not enabled yet ! ")
        quit()
        #x1dict = {    
        #        "type": "evolving_gaussian",
        #        "mu_func": mu_linear,
        #        "mu_param_names": ["slope"],
        #        "mu_param_values": [x1_slope],
        #        "sigma": x1_std
        #}

    return x1dict 


def get_args():
    
    
    parser = argparse.ArgumentParser()

    msg = """
    Filepath to the CSV file containing the output parameters  """
    parser.add_argument("--FILE", help=msg, type=str)

    msg = """
    Filepath to the config file that generated the relevant model parameters """
    parser.add_argument("--CONFIG", help=msg, type=str)


    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    infos = load_kestrel(args.CONFIG)
    try:
        infos['Splits']
    except KeyError:
        infos['Splits'] = {}

    try:
        infos['selection_parameters']
    except KeyError:
        infos['selection_parameters'] = {}

    dicts = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations'], infos['selection_parameters']]

    #load in the csv file 
    if not args.FILE.endswith(".CSV"):
        raise ValueError(
            f"Sorry, but your filename is required to end with CSV. "
            f"You gave me {args.FILE}!"
        )

    param_dict = read_params(args.FILE)
    param_labels = ["SALT2c", "RV", "SALT2x1","EBV",]

    host_bins = np.arange(6,14,0.1)
    c_bins = np.arange(-0.5,0.51,0.01)
    RV_bins = np.arange(1.2,8.1,0.1)
    EBV_bins = np.arange(0,1, 0.01)
    x1_bins = np.arange(-5,5.1, 0.1)

    arrays_list = [
        c_bins,
        (RV_bins, host_bins) if infos['Splits'].get('SIM_RV', False) else RV_bins,
        (x1_bins, host_bins) if infos['Splits'].get('SIM_x1', False) else x1_bins,
        (EBV_bins, host_bins) if infos['Splits'].get('SIM_EBV', False) else EBV_bins,
    ]

    print("Mixture Models not currently enabled...")
    labels = unspool_labels(infos['param_names'], dicts, 
                            infos['Latex_Names'], infos['Functions'], mixture=False, infos=infos )

    corr_bins = [0,10,15]

    pdf_entries = [
        # colour
        construct_dict_colour(infos, param_dict),
        construct_dict_RV(infos, param_dict),
        construct_dict_x1(infos, param_dict), 
        construct_dict_ebv(infos, param_dict),    
    ]

    GENPDF_WRITER(
    pdf_entries=pdf_entries,
    arrays_list=arrays_list,
    filename=args.FILE.replace("CSV","GENPDF"),
    param_labels=param_labels,
    param_dict=param_dict,
    corr_bins=corr_bins,
    beta=[param_dict[r'$\beta_{int}$ $\mu$'], param_dict[r'$\beta_{int}$ $\sigma$']],
    sigint=param_dict[r'$\sigma_{\rm int}$']
)