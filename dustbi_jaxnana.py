import jax 


##########################
## Simulator Functions 
##########################

def compile_simulator(sim_input, infos, n_sne):

    import jax_snlc_sim.run as sim_run

    conditional_samplers, direct_samplers = assign_samplers(infos)

    simulate = sim_run.build_sbi_pipeline(
        sim_input,
        n_sne=n_sne,
        samplers=direct_samplers,
        conditional_samplers=conditional_samplers
    )
    
    return simulate, (conditional_samplers, direct_samplers)

def JAX_Params(priors, samplers, infos):
    """Generates parameters to pass to simulator object.
    """
    params, prior_draw = assign_params_to_jaxnana(priors, samplers,  infos)

    return params

def JAX_Simulate(key, param, simulate):
    """Simulates with fixed set of params from assign_params_to_jaxnana
    """
    return simulate(key,params=param)

##########################
## For Multiple Populations 
##########################

def generate_two_pops(simulator, priors, shared_params, NSNe, infos):

    """
    Creates a simulation with a mixture model. Right now you cannot deliberately specify the priors or mixing fraction.
    This is currently handled inside the function automatically, but it is returned as a byproduct of the function. 
    """
    
    priors1 = JAX_Params(priors, samplers, infos)
    priors2 = JAX_Params(priors, samplers, infos)
 
    shared_params.append('a')

    def clean_draw(draw):
        draw = jnp.asarray(draw)
        return jnp.squeeze(draw) if draw.size == 1 else draw
    

    a = clean_draw(priors['a'].sample())
    
    theta_pop = {}
    
    for param in priors1.keys():
        if param not in shared_params:
            theta_pop[param+"_A"] = priors1[param]
            theta_pop[param+"_B"] = priors2[param]
    
    for param in priors1.keys():
        if param in shared_params:
            theta_pop[param] = priors1[param]
            priors2[param]   = priors1[param]

    results1, selection1 = JAX_Simulate(jax.random.PRNGKey(0), priors1, simulator)
    results2, selection2 = JAX_Simulate(jax.random.PRNGKey(0), priors2, simulator)
    
    theta_pop['a'] = a
    a = int(a*NSNe)
    
    results, selection = replace_random_sne(
        results1,
        results2,
        selection1,
        selection2,
        a,
    )

    return results, selection, theta_pop

#####

def replace_random_sne(
    results, results1,
    selection, selection1,
    N,
    seed=42
):

    """
    complicated function that exists to merge two outputs of (result, selection)  
    from JAXNANA into one singular one with the correct mixing ratio. 
    """

    import jax.numpy as jnp
    import random

    # Get number of SNe
    n_sne = results["epoch_count"].shape[0]

    if N > n_sne:
        raise ValueError(f"Only {n_sne} SNe available.")

    # Select N random SNe once, for both dictionaries
    rng = random.Random(seed)
    indices = jnp.array(rng.sample(range(n_sne), N))

    # Recursively replace selected SNe
    def recurse(target, source, path=""):

        if set(target.keys()) != set(source.keys()):
            raise ValueError(f"Dictionary keys differ at {path}")

        for key in target:
            name = f"{path}.{key}" if path else key

            if isinstance(target[key], dict):
                if not isinstance(source[key], dict):
                    raise TypeError(f"Structure mismatch at {name}")

                recurse(target[key], source[key], name)

            elif target[key] is None:
                if source[key] is not None:
                    raise ValueError(f"None mismatch at {name}")

            else:
                if source[key] is None:
                    raise ValueError(f"None mismatch at {name}")

                if target[key].shape != source[key].shape:
                    raise ValueError(f"Shape mismatch at {name}")

                target[key] = target[key].at[indices].set(
                    source[key][indices]
                )

    # Update both dictionaries using identical indices
    recurse(results, results1, "results")
    recurse(selection, selection1, "selection")

    return results, selection


##########################
## IO Functions
##########################

def assign_samplers(infos):

    #A bit of a hack here; we always expect 2 theta but only ever pass one. 
    #Make assign_params_to_jaxnana much cleaner down the line. 
    def _extract_correlation(function, correlation_name):
        def extract_correlation(key, event, theta, correlation=None):
            corr = event[correlation_name]
            return function(key, theta, corr)

        return extract_correlation
    
    conditionals = {} ; directs = {} 

    for parameter in infos['param_names']+infos['nuisance_parameters']:
        if parameter == "2POPS":
            continue 

        if parameter in infos['Splits']:
            split_type = infos['Splits'][parameter][0]

            if split_type == 'Stepwise':
                print(f"Assigning {parameter} as a split function with {infos['Splits'][parameter][1]} at {infos['Splits'][parameter][2]}, with {infos['Functions'][parameter].__name__}")

                correlated_parameter = infos['Splits'][parameter][1]
                correlated_value = infos['Splits'][parameter][2]

                conditionals[parameter] = jax.jit(
                    make_vulgar_split(infos['Functions'][parameter], correlated_parameter, correlated_value)
                )
            
            else:
                raise NotImplementedError(f"Split type {split_type} not implemented")

        elif parameter in infos['Correlations'].keys():
                print(f"Assigning {parameter} to correlate with {infos['Correlations'][parameter]} as {infos['Functions'][parameter].__name__}")
                conditionals[parameter] = jax.jit(_extract_correlation(infos['Functions'][parameter], infos['Correlations'][parameter]))
        

        else:
            print(f"Assigning {parameter} as {infos['Functions'][parameter].__name__}")
            directs[parameter] = jax.jit(infos['Functions'][parameter])

    return conditionals, directs



def make_vulgar_split(function, correlated_parameter, correlated_value):
    import jax.numpy as jnp

    def vulgar_split(
        key, event,
        theta=None,
        theta1=None,
        theta2=None,
        correlation=None
    ):
        event_value = event[correlated_parameter]

        if theta is not None:
            corr = (event_value, correlated_value)
            return function(key, theta, corr)

        condition = event_value > correlated_value
        selected_theta = jnp.where(condition, theta1, theta2)

        return function(key, selected_theta, correlation)

    return vulgar_split


def assign_params_to_jaxnana(priors, samplers, infos):
    import jax.numpy as jnp

    conditional, direct = samplers
    params = {}
    prior_draw = {}

    def clean_draw(draw):
        draw = jnp.asarray(draw)
        return jnp.squeeze(draw) if draw.size == 1 else draw

    # Direct parameters
    for key in direct:
        pdraw = clean_draw(priors[key].sample())
        prior_draw[key] = pdraw
        params[key] = {
            "theta": pdraw,
            "correlation": None
        }

    # Conditional parameters
    for key in conditional:

        theta1 = clean_draw(priors[key].sample())

        if key in infos['Correlations'].keys():
            params[key] = {"theta": theta1, "correlation": None}
            prior_draw[key] = {"theta": theta1}

        elif key in ["gammaDM", ]:
            params[key] = {"theta": theta1}
            prior_draw[key] = {"theta": theta1}

        else:        

            theta2 = clean_draw(priors[key].sample())
            prior_draw[key] = {
                "theta1": theta1,
                "theta2": theta2
            }
            params[key] = {
                "theta1": theta1,
                "theta2": theta2,
                "correlation": None
            }

    return params, prior_draw

