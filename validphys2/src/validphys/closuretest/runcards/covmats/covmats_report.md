%Closure test: missing systematics in covmats.

Summary
-------

This report shows {@ fit @}'s covmats.
Below is a summary of the main closure test features.

{@ closuretest_summary @}

Covariance matrices
-------------------

Below are displayed the original and manipulated covariance matrices used in fits. Note that level 1 shifts are always produced with the right experimental covmat.

### Sampling covmat

This matrix is used to generate a multivariate Gaussian distribution to be sampled from in order to generate pseudodata shifts.

{@ plot_sampling_covmats_comparison @}
{@ plot_difference_original_used_sampling_covmat @}

### Fitting covmat

This matrix is used to calculate $\chi^2$ in the loss function.

{@ plot_fitting_covmats_comparison @}
{@ plot_difference_original_used_fitting_covmat @}

Normalized diagonal
-------------------

Here, the square root of the diagonal elements of the covmats normalized to data are plotted.

### Sampling covmat

{@ plot_diagonal_vs_used_sampling_covmat @}

### Fitting covmat

{@ plot_diagonal_vs_used_fitting_covmat @}
