%Closure test: missing systematics in covmats.

Summary
-------

This report shows {@ fit @}'s covmats.
Below is a summary of the main closure test features.

{@ closuretest_summary @}

Covariance matrices
-------------------

Below are displayed the covariance matrices used in fits. Note that level 1 shifts are always produced with the right experimental covmat.

### Sampling covmat

This matrix is used to generate a multivariate Gaussian distribution to be sampled from in order to generate pseudodata shifts.

{@ plot_sampling_covmat_heatmap @}

### Fitting covmat

This matrix is used to calculate $\chi^2$ in the loss function.

{@ plot_fitting_covmat_heatmap @}

Normalized diagonal
-------------------

Here, the square root of the diagonal elements of the covmats normalized to data are plotted.

{@ plot_diagonal_sampling_fitting_covmat @}