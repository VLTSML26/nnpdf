import pandas as pd
import numpy as np
import scipy.linalg as la
from validphys.api import API
from sklearn.preprocessing import StandardScaler

INTRA_DATASET_SYS_NAME = ("UNCORR", "CORR", "THEORYUNCORR", "THEORYCORR")

def main():
    dsinps = [
        {'dataset': 'CHORUSNUPb_dw_ite', 'frac': 0.75},
        {'dataset': 'CHORUSNBPb_dw_ite', 'frac': 0.75},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'frac': 0.75, 'cfac': ['MAS']},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'frac': 0.75, 'cfac': ['MAS']},
        {'dataset': 'NMCPD_dw_ite', 'frac': 0.75},
        {'dataset': 'BCDMSD_dw_ite', 'frac': 0.75},
        {'dataset': 'SLACD_dw_ite', 'frac': 0.75},
        {'dataset': 'NMC', 'frac': 0.75},
        {'dataset': 'SLACP_dwsh', 'frac': 0.75},
        {'dataset': 'BCDMSP_dwsh', 'frac': 0.75},
        {'dataset': 'HERACOMBNCEM', 'frac': 0.75},
        {'dataset': 'HERACOMBNCEP460', 'frac': 0.75},
        {'dataset': 'HERACOMBNCEP575', 'frac': 0.75},
        {'dataset': 'HERACOMBNCEP820', 'frac': 0.75},
        {'dataset': 'HERACOMBNCEP920', 'frac': 0.75},
        {'dataset': 'HERACOMBCCEM', 'frac': 0.75},
        {'dataset': 'HERACOMBCCEP', 'frac': 0.75},
        {'dataset': 'HERACOMB_SIGMARED_C', 'frac': 0.75},
        {'dataset': 'HERACOMB_SIGMARED_B', 'frac': 0.75}
    ]
    inp = dict(dataset_inputs=dsinps, theoryid=200, use_cuts="internal")
    dataset_inputs_loaded_cd_with_cuts = API.dataset_inputs_loaded_cd_with_cuts(**inp)
    data_input = API.data_input(**inp)
    tot_trace = np.trace(API.dataset_inputs_covmat_from_systematics(**inp))
    missingsys_exps = [
        'CHORUSNUPb_dw_ite',
        'CHORUSNBPb_dw_ite',
        'NTVNBDMNFe_dw_ite',
        'NTVNUDMNFe_dw_ite'
    ]
    traces_value = r'% Covmat trace changed'
    norms_value = "Normalized sys's norms (distance to mean in terms of sigma)"
    traces_dfs = []
    norms_dfs = []

    for cd in dataset_inputs_loaded_cd_with_cuts:
        if cd.setname in missingsys_exps:
            traces_list = []
            normalized_to_data_norms = []
            sys_errors = cd.systematic_errors()
            for key in sys_errors.keys():
                covmat = get_covmat(key, dataset_inputs_loaded_cd_with_cuts, data_input)
                trace = np.trace(covmat)
                traces_list += [
                    np.abs(trace - tot_trace) * 100 / tot_trace
                ]
                relative_errors = sys_errors[key].values / cd.central_values.values
                normalized_to_data_norms += [relative_errors @ relative_errors]
            traces_dict = {
                (cd.setname, sys_errors.keys()[i]): trace
                for i, trace in enumerate(traces_list)
            }
            traces_df = pd.DataFrame.from_dict(traces_dict, orient='index', columns=[traces_value])
            traces_df.sort_values(by=traces_value, inplace=True, ascending=False)
            traces_dfs += [traces_df]
            normalized_to_data_norms = np.asarray(normalized_to_data_norms).reshape(-1, 1)
            normalized_to_data_norms = StandardScaler().fit_transform(normalized_to_data_norms)
            norms_dict = {
                (cd.setname, sys_errors.keys()[i]): np.abs(norm)
                for i, norm in enumerate(normalized_to_data_norms)
            }
            norms_df = pd.DataFrame.from_dict(norms_dict, orient='index', columns=[norms_value])
            norms_df.sort_values(by=norms_value, inplace=True, ascending=False)
            norms_dfs += [norms_df]
    
    overall_traces_df = pd.concat(traces_dfs)
    overall_traces_df.index = pd.MultiIndex.from_tuples(overall_traces_df.index)
    print(overall_traces_df.groupby(level=0).head(10))
    overall_norms_df = pd.concat(norms_dfs)
    overall_norms_df.index = pd.MultiIndex.from_tuples(overall_norms_df.index)
    print(overall_norms_df.groupby(level=0).head(10))
        
def construct_covmat(
    stat_errors: np.array,
    sys_errors: pd.DataFrame
):
    diagonal = stat_errors ** 2
    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    diagonal += (sys_errors.loc[:, is_uncorr].to_numpy() ** 2).sum(axis=1)

    corr_sys_mat = sys_errors.loc[:, ~is_uncorr].to_numpy()
    return np.diag(diagonal) + corr_sys_mat @ corr_sys_mat.T

def get_covmat(
    key,
    dataset_inputs_loaded_cd_with_cuts,
    data_input,
    use_weights_in_covmat=True,
):
    special_corrs = []
    block_diags = []
    weights = []

    for cd, dsinp in zip(dataset_inputs_loaded_cd_with_cuts, data_input):
        stat_errors = cd.stat_errors.to_numpy()
        sys_errors = cd.systematic_errors()
        weights.append(np.full_like(stat_errors, dsinp.weight))
        sys_errors[key] = 0.
        is_intra_dataset_error = sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        block_diags.append(construct_covmat(
            stat_errors,
            sys_errors.loc[:, is_intra_dataset_error]
        ))
        special_corrs.append(sys_errors.loc[:, ~is_intra_dataset_error])

    special_sys = pd.concat(special_corrs, axis=0, sort=False)
    special_sys.fillna(0, inplace=True)

    diag = la.block_diag(*block_diags)
    covmat = diag + special_sys.to_numpy() @ special_sys.to_numpy().T

    if use_weights_in_covmat:
        sqrt_weights = np.sqrt(np.concatenate(weights))
        covmat = (covmat / sqrt_weights).T / sqrt_weights
    
    return covmat

if __name__ == '__main__':
    main()