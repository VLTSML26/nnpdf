import pandas as pd
import numpy as np
from validphys.api import API

INTRA_DATASET_SYS_NAME = ("UNCORR", "CORR", "THEORYUNCORR", "THEORYCORR")

def main():
    dsinps = [
        {'dataset': 'CHORUSNUPb_dw_ite', 'frac': 0.75},
        {'dataset': 'CHORUSNBPb_dw_ite', 'frac': 0.75}
    ]
    inp = dict(dataset_inputs=dsinps, theoryid=200, use_cuts="internal")
    dataset_inputs_loaded_cd_with_cuts = API.dataset_inputs_loaded_cd_with_cuts(**inp)
    data_input = API.data_input(**inp)

    features_dict = {}
    for cd, dsinp in zip(
        dataset_inputs_loaded_cd_with_cuts,
        data_input,
    ):
        sys_errors = cd.systematic_errors()
        is_intra_dataset_error = sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        proxy_random_array = np.random.rand(sys_errors.shape[0])
        intra_dataset_matrices_norm = []
        not_intra_dataset_matrices_norm = []
        sys_norms_random = []
        sys_squared_norms = []

        for key in sys_errors.keys():
            
            # FIRST CRITERION: get S @ S.T by selecting out a systematic
            proxy_sys_errors = sys_errors.copy(deep=True)
            proxy_sys_errors[key] = 0.
            intra_dataset_corr = get_corrsysmat(proxy_sys_errors.loc[:, is_intra_dataset_error])
            not_intra_dataset_corr = get_corrsysmat(proxy_sys_errors.loc[:, ~is_intra_dataset_error])
            intra_dataset_matrices_norm += [np.linalg.norm(intra_dataset_corr)]
            not_intra_dataset_matrices_norm += [np.linalg.norm(not_intra_dataset_corr)]

            # SECOND CRITERION: get column norm by product with proxy random array
            sys_norms_random += [proxy_random_array @ sys_errors[key].to_numpy()]

            # THIRD CRITERION: get regular squared norm
            sys_squared_norms += [sys_errors[key].to_numpy() @ sys_errors[key].to_numpy()]

        # FIRST CRITERION: get min and argmin
        intra_dataset_min = np.min(intra_dataset_matrices_norm)
        intra_dataset_argmin = sys_errors[sys_errors.keys()[np.argmin(intra_dataset_matrices_norm)]].name
        not_intra_dataset_min = np.min(not_intra_dataset_matrices_norm)
        not_intra_dataset_argmin = sys_errors[sys_errors.keys()[np.argmin(not_intra_dataset_matrices_norm)]].name
        # SECOND CRITERION: get max and argmax
        max_norms_random = np.max(sys_norms_random)
        argmax_norms_random = sys_errors[sys_errors.keys()[np.argmax(sys_norms_random)]].name
        # THIRD CRITERION: get max and argmax
        max_norms = np.max(sys_squared_norms)
        argmax_norms = sys_errors[sys_errors.keys()[np.argmax(sys_squared_norms)]].name
        # FOURTH CRITERION: get max and argmax of sum of elements
        max_sums = sys_errors.sum(axis=0).abs().max()
        argmax_sums = sys_errors.sum(axis=0).abs().idxmax()
        
        # Update features dict
        features_dict[
            (dsinp.name, 'Min corrmat', 'Inside datsaet')
        ] = (intra_dataset_min, intra_dataset_argmin)
        features_dict[
            (dsinp.name, 'Min corrmat', 'Across datasets')
        ] = (not_intra_dataset_min, not_intra_dataset_argmin)
        features_dict[
            (dsinp.name, 'Max sys', 'Random rotation')
        ] = (max_norms_random, argmax_norms_random)
        features_dict[
            (dsinp.name, 'Max sys', 'L2 norm')
        ] = (max_norms, argmax_norms)
        features_dict[
            (dsinp.name, 'Max sys', 'Sum of elements (max of abs)')
        ] = (max_sums, argmax_sums)

    df = pd.DataFrame.from_dict(features_dict, orient='index', columns=['Value', 'Sys'])
    df.index = pd.MultiIndex.from_tuples(df.index)
    print(df)
    return

def get_corrsysmat(
    sys_errors: pd.DataFrame
):
    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    corr_sys_mat = sys_errors.loc[:, ~is_uncorr].to_numpy()
    return corr_sys_mat @ corr_sys_mat.T

if __name__ == '__main__':
    main()