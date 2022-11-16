import pandas as pd
import numpy as np
from validphys.api import API
from sklearn.preprocessing import StandardScaler

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
        global sys_errors
        sys_errors = cd.systematic_errors()
        is_intra_dataset_error = sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        proxy_random_array = np.random.rand(sys_errors.shape[0])
        intra_dataset_matrices_norm = []
        not_intra_dataset_matrices_norm = []
        sys_norms_random = []
        sys_squared_norms = []

        tot_intra_norm = np.linalg.norm(get_corrsysmat(sys_errors.loc[:, is_intra_dataset_error]))
        tot_not_intra_norm = np.linalg.norm(get_corrsysmat(sys_errors.loc[:, ~is_intra_dataset_error]))
        for key in sys_errors.keys():
            
            # FIRST CRITERION: get S @ S.T by selecting out a systematic
            intra_norm, not_intra_norm = get_corrmatnorm(
                key,
                is_intra_dataset_error,
                sys_errors
            )
            intra_dataset_matrices_norm += [
                np.abs(intra_norm - tot_intra_norm) * 100. / tot_intra_norm
            ]
            not_intra_dataset_matrices_norm += [
                np.abs(not_intra_norm - tot_not_intra_norm) * 100. / tot_not_intra_norm
            ]

            # SECOND CRITERION: get column norm by product with proxy random array
            sys_norms_random += [proxy_random_array @ sys_errors[key].to_numpy()]

            # THIRD CRITERION: get regular squared norm
            sys_squared_norms += [sys_errors[key].to_numpy() @ sys_errors[key].to_numpy()]

        # FIRST CRITERION: get min and argmin
        intra_dict = {
            sys_errors.keys()[i]: norm
            for i, norm in enumerate(intra_dataset_matrices_norm)
        }
        not_intra_dict = {
            sys_errors.keys()[i]: norm
            for i, norm in enumerate(not_intra_dataset_matrices_norm)
        }
        value = r'% Corrmat norm changed'
        intra_df = pd.DataFrame.from_dict(intra_dict, orient='index', columns=[value])
        intra_df.sort_values(by=value, inplace=True, ascending=False)
        not_intra_df = pd.DataFrame.from_dict(not_intra_dict, orient='index', columns=[value])
        not_intra_df.sort_values(by=value, inplace=True, ascending=False)
        # SECOND CRITERION: get max and argmax
        sys_norms_random = np.asarray(sys_norms_random).reshape(-1, 1)
        sys_norms_random = StandardScaler().fit_transform(sys_norms_random)
        random_dict = {
            sys_errors.keys()[i]: np.abs(randnorm)
            for i, randnorm in enumerate(sys_norms_random)
        }
        value = 'Abs distance from mean in terms of std'
        random_df = pd.DataFrame.from_dict(random_dict, orient='index', columns=[value])
        random_df.sort_values(by=value, inplace=True, ascending=False)
        # THIRD CRITERION: get max and argmax
        sys_squared_norms = np.asarray(sys_squared_norms).reshape(-1, 1)
        sys_squared_norms = StandardScaler().fit_transform(sys_squared_norms)
        sqrt_dict = {
            sys_errors.keys()[i]: np.abs(sqrnorm)
            for i, sqrnorm in enumerate(sys_squared_norms)
        }
        value = 'Abs distance from mean in terms of std'
        sqrt_df = pd.DataFrame.from_dict(sqrt_dict, orient='index', columns=[value])
        sqrt_df.sort_values(by=value, inplace=True, ascending=False)
        # # THIRD CRITERION: get max and argmax
        # max_norms = np.max(sys_squared_norms)
        # argmax_norms = sys_errors[sys_errors.keys()[np.argmax(sys_squared_norms)]].name
        # FOURTH CRITERION: get max and argmax of sum of elements
        max_sums = sys_errors.sum(axis=0).abs().max()
        argmax_sums = sys_errors.sum(axis=0).abs().idxmax()
        
        # Update features dict
        features_dict[
            (dsinp.name, 'Min corrmat', 'Inside datsaet')
        ] = (intra_df.iloc[0].values[0], intra_df.iloc[0].name)
        features_dict[
            (dsinp.name, 'Min corrmat', 'Across datasets')
        ] = (not_intra_df.iloc[0].values[0], not_intra_df.iloc[0].name)
        features_dict[
            (dsinp.name, 'Max sys', 'Random rotation')
        ] = (random_df.iloc[0].values[0], random_df.iloc[0].name)
        features_dict[
            (dsinp.name, 'Max sys', 'L2 norm')
        ] = (sqrt_df.iloc[0].values[0], sqrt_df.iloc[0].name)
        # features_dict[
        #     (dsinp.name, 'Max sys', 'L2 norm')
        # ] = (max_norms, argmax_norms)
        features_dict[
            (dsinp.name, 'Max sys', 'Sum of elements (max of abs)')
        ] = (max_sums, argmax_sums)
        print(
            '\n\n=====',
            dsinp.name,
            '\n\n1) S@S.T norm\n\n',
            ' 1a) Inside dataset\n',
            intra_df.head(),
            '\n\n 1b) Across other datasets\n',
            not_intra_df.head(),
            '\n\n2) Random proxy\n',
            random_df.head(),
            '\n\n3) Squared norm\n',
            sqrt_df.head()
        )

    df = pd.DataFrame.from_dict(features_dict, orient='index', columns=['Value', 'Sys'])
    df.index = pd.MultiIndex.from_tuples(df.index)
    print('\n\nSUMMARY TABLE\n', df)
    return

def get_corrsysmat(
    sys_errors: pd.DataFrame
):
    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    corr_sys_mat = sys_errors.loc[:, ~is_uncorr].to_numpy()
    return corr_sys_mat @ corr_sys_mat.T

def get_corrmatnorm(
    key,
    is_intra,
    sys_errors: pd.DataFrame,
):
    sys = sys_errors.copy(deep=True)
    sys[key] = 0.
    intra_norm = np.linalg.norm(get_corrsysmat(sys.loc[:, is_intra]))
    not_intra_norm = np.linalg.norm(get_corrsysmat(sys.loc[:, ~is_intra]))
    return intra_norm, not_intra_norm

if __name__ == '__main__':
    main()