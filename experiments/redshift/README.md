

##Quasar Photo-Spec Model Experiments

##TODO: move these experiments to their own repository 

#### Roadmap

1. Fit a basis to BOSS data using Monte Carlo EM
  ```
  python preproc_fit_basis_mcem.py -K 6 -N 2000 -S redshift
  ```
  where
  * `-K`: number of bases
  * `-S`: split type (redshift, flux, redshift)
  * `-N`: number of training examples to use
  this will create files in `mcem_fits` that include the basis, input location (wavelengths) that they correspond to and some other fit metadata (files look like `mcem_fits/qso_basis_K_6_split_flux.pkl`).

  This routine requires the `FNMF/BayesNMF` (will be released soon) package for fitting functional matrix factorization models and the `SpecExperiments` package, which ought to be merged w/ this repository. 
  

2. Fit prior to the training sample weights from the first step 
  ```
  python preproc_fit_weight_prior_mcem.py
  ```
  which goes in and grabs the sample from each fit file in `mcem_fits/` and cross validates a mixture of gaussians fit on the logit-space of the normalized weights.  It also pulls in a few thousand more validation example from the corresponding data-split's validation data, which is used to get a better idea of the distribution.  Output GMM fit files look like `mcem_fits/prior_weight_K_6_split_flux.pkl`. 

3. Fit an individual photometric observation (and compare it to just naive slice sampler) for a model conditioned on the map basis
  ```
  python sample_weight_comparison.py
  ```
  which will run a few chains of parallel-tempering + slice sampling and then a few chains of just slice sampling. It will then create a few traceplots, histogram comparisons, and a look at convergence diagnostics (r-hat, n_eff, ess). 
  
4. Fit a big sample of photometric observations
  ```
  python sample_weights_redshift_temper_mcem.py 
  ```
  which should 
  TODO: use Joblib to run a bunch of these in parallel from within the same python job. 

  
