#!/bin/sh
echo "taskid is: " $TF_TASKID 

# DEBUGGING
##PBS_O_WORKDIR=`exec pwd`
##TF_TASKID=1

# MCMC params
test_n=${TF_TASKID}
Nsamps=8000
Nchains=8
LAM_SUBSAMPLE=10
NUM_BASES=4
SPLIT_TYPE="redshift"    #"random", "flux", "redshift"
BASIS_DIR=/global/homes/a/acmiller/Proj/DESIMCMC/experiments/redshift/cache/basis_fits
echo "Running test_n=${test_n},  ${Nsamps}-chain MCMC in ${PBS_O_WORKDIR}, from ${BASIS_DIR}"

# my output files
mkdir -p ${PBS_O_WORKDIR}/outfiles
mkdir -p ${PBS_O_WORKDIR}/samps
OUT_DIR=${PBS_O_WORKDIR}/samps

python ../../sample_weights_redshift_temper.py \
  ${test_n} \
  ${Nsamps} \
  ${Nchains} \
  ${LAM_SUBSAMPLE} \
  ${NUM_BASES} \
  ${SPLIT_TYPE} \
  ${OUT_DIR} \
  ${BASIS_DIR} &> ${PBS_O_WORKDIR}/outfiles/samp.${TF_TASKID}.out

