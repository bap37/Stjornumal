from dustbi_simulator import *
from Functions import *
from dustbi_nn import PopulationEmbeddingFull
import argparse
import torch
import torch.nn as nn
from sbi.neural_nets import classifier_nn
from sbi.inference import NRE_A




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CONFIG", help="Configuration yaml for NRE model comparison.", type=str)
    parser.add_argument("--BIRD", help="Prints a nice bird :)", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    if args.BIRD:
        print("I'm very sorry but the kestrel hasn't taken flight yet!")
        quit()

    if not args.CONFIG:
        print("No configuration file provided via --CONFIG. Quitting.")
        quit()

    infos = load_kestrel(args.CONFIG)

    datfilename = infos['Data_File'][0]
    simfilename = infos['Simbank_File'][0]
    parameters_to_condition_on = infos['parameters_to_condition_on']
    ndim = len(parameters_to_condition_on)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, dfdata = load_data(simfilename, datfilename)

    num_simulations = infos['sim_parameters']['n_sim']

    # Compute MURES for sim bank and data
    output_distribution = preprocess_input_distribution(
        df, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    df['MURES'] = add_distance(output_distribution)

    output_distribution = preprocess_input_distribution(
        dfdata, parameters_to_condition_on[:-1] + ['x0', 'x0ERR', 'MU'])
    dfdata['MURES'] = add_distance(output_distribution)

    # Observed data — shape (n_sne, ndim) -> unsqueeze to (1, n_sne, ndim)
    x_obs = preprocess_data(parameters_to_condition_on, dfdata).unsqueeze(0)

    # --- Nominal model (model 1) ---
    dicts_1 = [infos['Functions'], infos['Splits'], infos['Priors'], infos['Correlations']]
    param_names_1 = infos['param_names']
    params_to_fit_1 = parameter_generation(param_names_1, dicts_1)
    layout_1 = build_layout(params_to_fit_1, dicts_1)

    mixture_1 = 'Population_B' in infos
    split_positions_1 = None
    if mixture_1:
        pop_b = infos['Population_B']
        shared_params_1 = [p for p in pop_b.get('shared_params', []) if p not in ('STEP', 'SCATTER')]
        split_names_1 = [n for n in param_names_1 if n not in shared_params_1 and n not in ('STEP', 'SCATTER')]
        dicts_1B = [infos['Functions'], infos['Splits'], pop_b['Priors'], infos['Correlations']]
        priors_A = build_distribution_priors(param_names_1, dicts_1)
        priors_B_split = build_distribution_priors(split_names_1, dicts_1B)
        mix = pop_b['mixing_prior']
        f_prior = BoxUniform(low=torch.tensor([mix[0]]), high=torch.tensor([mix[1]]))
        special = build_special_priors(param_names_1, dicts_1)
        priors_1 = MultipleIndependent(priors_A + priors_B_split + [f_prior] + special)
        split_positions_1 = compute_split_positions(layout_1, shared_params_1)
        assert len(split_positions_1) == len(priors_B_split)
    else:
        priors_1 = prior_generator(param_names_1, dicts_1)

    nominal_sim = make_batched_simulator(
        layout_1, df, param_names_1, parameters_to_condition_on,
        dicts_1, dfdata, sub_batch=500, device=device, mixture=mixture_1,
        split_positions=split_positions_1
    )

    print(f"Simulating {num_simulations} from nominal model ({args.CONFIG})...")
    theta_1 = priors_1.sample((num_simulations,)).to(device)
    x1 = nominal_sim(theta_1).cpu()

    # NaN mask for model 1 (constant across comparisons)
    mask1 = torch.isfinite(x1).all(dim=(1, 2))
    x1_clean = torch.nan_to_num(x1, nan=999999999)
    print(f"  {args.CONFIG}: {x1_clean.shape[0]} valid / {x1.shape[0]} total")

    classifier1 = classifier_nn(
        model="resnet",
        embedding_net_x=PopulationEmbeddingFull(input_dim=ndim),
    )

    inference1 = NRE_A(
        prior=priors_1,
        classifier=classifier1,
    )

    ratio_estimator1 = (
        inference1
        .append_simulations(theta_1, x1_clean)
        .train()
    )


    # --- Compare against each model ---
    for model_path in infos['Models_Comparison']:
        print(f"\n{'='*60}")
        print(f"Comparing {args.CONFIG} vs {model_path}")
        print(f"{'='*60}")

        comp_infos = load_kestrel(model_path)

        try: 
            comp_infos['Splits']
        except KeyError:
            comp_infos['Splits'] = {}
            print("Temporarily hacking splits to be an empty dict")
        

        dicts_2 = [comp_infos['Functions'], comp_infos['Splits'],
                    comp_infos['Priors'], comp_infos['Correlations']]
        param_names_2 = comp_infos['param_names']
        params_to_fit_2 = parameter_generation(param_names_2, dicts_2)
        layout_2 = build_layout(params_to_fit_2, dicts_2)

        mixture_2 = 'Population_B' in comp_infos
        split_positions_2 = None
        if mixture_2:
            pop_b2 = comp_infos['Population_B']
            shared_params_2 = [p for p in pop_b2.get('shared_params', []) if p not in ('STEP', 'SCATTER')]
            split_names_2 = [n for n in param_names_2 if n not in shared_params_2 and n not in ('STEP', 'SCATTER')]
            dicts_2B = [comp_infos['Functions'], comp_infos['Splits'], pop_b2['Priors'], comp_infos['Correlations']]
            priors_2A = build_distribution_priors(param_names_2, dicts_2)
            priors_2B_split = build_distribution_priors(split_names_2, dicts_2B)
            mix2 = pop_b2['mixing_prior']
            f_prior_2 = BoxUniform(low=torch.tensor([mix2[0]]), high=torch.tensor([mix2[1]]))
            special_2 = build_special_priors(param_names_2, dicts_2)
            priors_2 = MultipleIndependent(priors_2A + priors_2B_split + [f_prior_2] + special_2)
            split_positions_2 = compute_split_positions(layout_2, shared_params_2)
            assert len(split_positions_2) == len(priors_2B_split)
        else:
            priors_2 = prior_generator(param_names_2, dicts_2)

        comp_sim = make_batched_simulator(
            layout_2, df, param_names_2, parameters_to_condition_on,
            dicts_2, dfdata, sub_batch=500, device=device, mixture=mixture_2,
            split_positions=split_positions_2
        )

        print(f"Simulating {num_simulations} from comparison model ({model_path})...")
        theta_2 = priors_2.sample((num_simulations,)).to(device)
        x2 = comp_sim(theta_2).cpu()

        mask2 = torch.isfinite(x2).all(dim=(1, 2))
        x2_clean = torch.nan_to_num(x2, nan=999999999)
        print(f"  {model_path}: {x2_clean.shape[0]} valid / {x2.shape[0]} total")

        classifier2 = classifier_nn(
            model="resnet",
            embedding_net_x=PopulationEmbeddingFull(input_dim=ndim),
        )

        inference2 = NRE_A(
            prior=priors_2,
            classifier=classifier2,
        )

        ratio_estimator2 = (
            inference2
            .append_simulations(theta_2, x2_clean)
            .train()
        )
        with torch.no_grad():


            N = 10000

            x_obs_batch = x_obs

            # ---- Model 1 ----

            theta_mc_1 = priors_1.sample((N,))
            x_rep_1 = x_obs_batch.expand(N, -1, -1)

            log_r1 = ratio_estimator1(theta_mc_1, x_rep_1)

            log_evidence1 = torch.logsumexp(log_r1, dim=0) - torch.log(
                torch.tensor(log_r1.shape[0], dtype=torch.float32)
            )

            # ---- Model 2 ----
            theta_mc_2 = priors_2.sample((N,))
            x_rep_2 = x_obs_batch.expand(N, -1, -1)

            log_r2 = ratio_estimator2(theta_mc_2, x_rep_2)


            log_evidence2 = torch.logsumexp(log_r2, dim=0) - torch.log(
                torch.tensor(log_r2.shape[0], dtype=torch.float32)
            )

            # ---- Bayes factor (stay in log space!) ----
            log_bf_12 = log_evidence1 - log_evidence2

            print(log_bf_12)
            bf_12 = torch.exp(log_bf_12)
            print("Exponentiated Factor",bf_12)