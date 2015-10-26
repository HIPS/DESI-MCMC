

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
  this will create files in `mcem_fits` that include the basis and input location (wavelengths) that they correspond to.  

  This routine requires the `FNMF/BayesNMF` (will be released soon) package for fitting functional matrix factorization models and the `SpecExperiments` package, which ought to be merged w/ this repository. 

2. Fit prior to the training sample weights from the first step 
  ```
  python preproc_fit_weight_prior_mcem.py ...
  ```
  which should create some pickle files w/ the GMM fit. 
  TODO: incorporate way more data in this fit (sample w's conditioned on the basis for validation data)

3. Fit individual photometric observations to a model conditioned on the basis.  
  ```
  python sample_weights_redshift_temper_mcem.py 
  ```
  which should 
  TODO: use Joblib to run a bunch of these in parallel from within the same python job. 

4. Run a single slice sampler vs a parallel tempering slice sampler and watch it not mix
  ```
  python parallel_tempering_comparison.py
  ```
  selects out a single, multi-modal example and shows a slice sampler not mix super well.
  
  
