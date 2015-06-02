#!/bin/sh
echo "taskid is: " $TF_TASKID 

# DEBUGGING
#PBS_O_WORKDIR=`exec pwd`
#TF_TASKID=1

# MCMC params
test_n=${TF_TASKID}
Nsamps=8000
Nchains=8
LAM_SUBSAMPLE=10
NUM_BASES=4
SPLIT_TYPE="redshift"    #"random", "flux", "redshift"
BASIS_DIR=/global/homes/a/acmiller/Proj/DESIMCMC/experiments/redshift/cache/basis_locked
PRIOR_TYPE="mog"
echo "Running test_n=${test_n},  ${Nsamps}-chain MCMC in ${PBS_O_WORKDIR}, from ${BASIS_DIR}"

# my output files
mkdir -p ${PBS_O_WORKDIR}/outfiles
OUT_DIR=/project/projectdirs/das/quasar/${PBS_O_WORKDIR}/samps
mkdir -p ${OUT_DIR}

python ../../sample_weights_redshift_temper.py \
  ${test_n} \
  ${Nsamps} \
  ${Nchains} \
  ${LAM_SUBSAMPLE} \
  ${NUM_BASES} \
  ${SPLIT_TYPE} \
  ${OUT_DIR} \
  ${BASIS_DIR} \
  ${PRIOR_TYPE} &> ${PBS_O_WORKDIR}/outfiles/samp.${TF_TASKID}.out

