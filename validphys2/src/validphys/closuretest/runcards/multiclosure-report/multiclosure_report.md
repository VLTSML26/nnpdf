%NNPDF Multicosure Report Test

Summary 
-------

This report provides statistical indicators in order to determine the success of new closure inconsistency-tests.
The following table aims at showing all the features of the test by reading the Validphys runcard.

{@ closuretest_summary @}

The data have been produced by running the following closure fits.
Asterisks may appear alongside the fit name if that fit's parameters do not realize their expected values within the errorbars.
Any further consideration on such disagreements is left to the reader.

{@ multiclosure_summary @}

Click [here]({@ plot_replicas_report report @}) to see the fakepdf and the set from which it has been randomly extracted (default is NNPDF4.0).

$\Delta\chi^2$ by fit
---------------------

[Here]({@ plot_delta_chi2_report report @}) are displayed the $\Delta\chi^2$ distributions for each fit.

Closure test indicators
-----------------------

In this section we provide results on the main closure-test indicators: $R_\text{bv}$ and $\xi_{1\sigma}$. Here, the user will find informations about their total value and can navigate into seperate sections for single experiment's results.

### Bias-to-variance ratio

We show the normalized distributions of total bias and variance for this closure test. Different counts in one bin indicate the presence of non-gaussianity sources: should it happen here, plots for single experiments may provide more details.

{@ on_newdata plot_total_bias_variance_distributions @}

Below is shown the distribution of bootstrap samples of $R_\text{bv}$ alongside its fitted corresponding scaled normal distribution distribution for comparison.

{@ on_newdata plot_total_sqrt_ratio_bootstrap_distribution @}

Below is shown the trend of the total bootstrapped value of $R_\text{bv}$ as the number of fits considered increases. The errorbars are given by the bootstrap error.
The user may want to check if stability is reached and, if not, gather more data by running an additional number of fits.

{@ on_newdata plot_sqrt_ratio_behavior @}

For a more detailed discussion of $R_\text{bv}$ for single datasets or experiments, please refer to [this]({@ plot_biasvariance_report report@}) link.

### Quantile indicator

{@ on_newdata plot_experiments_xi_bootstrap_distribution @}
All the previous considerations hold.
The expected value for this indicator is calculated from the measured value of $R_\text{bv}$.

{@ on_newdata plot_xi_behavior @}

Distance from input-pdf in pdf space
------------------------------------

{@ plot_pdf_central_diff_histogram @}

Distance from central predictions in data space
-----------------------------------------------

{@ on_newdata plot_data_central_diff_histogram @}
