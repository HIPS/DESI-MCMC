import cPickle as pickle
import numpy as np
from string import Template
from glob import glob
import sys, os
sys.path.append('..')
import redshift_utils as ru

#s = Template('SELECT * FROM $table_name WHERE $condition')
#s.safe_substitute(table_name='users')

table_string = \
"""
\\begin{tabular}{ p{2.9cm}|ccc|ccc|ccc}
 & \multicolumn{3}{c}{MAE} & \multicolumn{3}{c}{MAPE} & \multicolumn{3}{c}{RMSE} \\\\
 \hline
 split & XD & NN & Spec & XD & NN & Spec & XD & NN & Spec \\\\
 \hline
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 %s \\\\
 \hline
\end{tabular}
"""
row_string = "$div & $xdmae & $NNmae & $specmae & $xdmape & $NNmape & $specmape & $xdrmse & $NNrmse & $specrmse "
def sub(string, **kwargs): 
    temp = Template(string)
    return temp.safe_substitute(**kwargs)

# error calc functions
def rmse(true, pred):
    return np.sqrt(np.mean((true-pred)**2))
def mape(true, pred):
    return np.mean(np.abs(true - pred)/np.abs(true))
def mae(true, pred):
    return np.mean(np.abs(true-pred))

# DR10 qso dataset and spec files
SPLIT_TYPE = "random"
qso_psf_flux, qso_psf_flux_ivar, qso_psf_mags, qso_z, spec_files, train_idx, test_idx = \
    ru.load_DR10QSO_train_test_idx(split_type = SPLIT_TYPE)

spec_files_array = np.array(spec_files)
def get_spec_ids(idx):
    specs = [os.path.splitext(os.path.basename(s))[0] for s in spec_files_array[idx]]
    return np.array(specs)

if __name__=="__main__":

    ## rows for data split
    exp_types = ["random",    "flux",    "redshift", 
                 "random 50", "flux 50", "redshift 50",
                 "random 90", "flux 90", "redshift 90"
                ]
    row_strings = [row_string]*len(exp_types)
    for i, exp_type in enumerate(exp_types):

        # do hi/lo redshift idx
        do_per     = False
        div_string = "%s (all)"%exp_type
        if len(exp_type.split(" ")) > 1: 
            exp_type, per = exp_type.split(" ")
            per      = int(per)
            do_per   = True

        # load MOG 
        mog_exp_file = "../analysis_mog/%s/results.pkl"%exp_type
        with open(mog_exp_file, 'rb') as handle:
            mog_dict  = pickle.load(handle)
            mog_specs = mog_dict['spec_ids']
            mog_preds = mog_dict['preds']
            mog_true  = mog_dict['z_true']
            if do_per:
                z_fifty    = np.percentile(mog_true, per)
                div_string = "%s ($z > %2.2f$)"%(exp_type, z_fifty)
                hi_idx = mog_true > z_fifty
                print div_string, hi_idx.sum()
                mog_preds = mog_preds[hi_idx]
                mog_true  = mog_true[hi_idx]

        # load bovy
        bovy_exp_file = glob("../analysis_bovy_reproduce/bovy_exp_SPLIT-%s_TRAIN-*_TEST-*_SEED-42_MAXGAUSS-20_WIDTH-0.2_UNITS-mags.pkl"%exp_type)[0]
        with open(bovy_exp_file, 'rb') as handle:
            bovy_dict  = pickle.load(handle)
            bovy_specs = get_spec_ids(bovy_dict['test_idx_sub'])
            overlap_sub = np.in1d(bovy_specs, mog_specs)
            print "BOVY overlap # = %d"%overlap_sub.sum()
            bovy_preds  = bovy_dict['preds_mean'][overlap_sub]
            bovy_true   = bovy_dict['z_test'][overlap_sub]
            if do_per: 
                hi_idx = bovy_true > z_fifty
                bovy_preds = bovy_preds[hi_idx]
                bovy_true  = bovy_true[hi_idx]

        # load brescia
        nn_rand_file = glob("../analysis_brescia_reproduce/brescia_exp_SPLIT-%s_TRAIN-*_*-all_SEED-42_NUM_LAYERS-4_WIDTH-100_UNITS-nanos.pkl"%exp_type)[0]
        with open(nn_rand_file, 'rb') as handle:
            nn_dict     = pickle.load(handle)
            overlap_sub = np.in1d(nn_dict['test_spec_ids'], mog_specs)
            print "NN overlap # = %d"%overlap_sub.sum()
            nn_preds = nn_dict['preds'][overlap_sub]
            nn_true  = nn_dict['z_test'][overlap_sub]
            if do_per:
                hi_idx = nn_true > z_fifty
                nn_preds = nn_preds[hi_idx]
                nn_true  = nn_true[hi_idx]

        # bold the best error in each case
        def format_errors(errfn):
            spec = errfn(mog_true, mog_preds)
            xd   = errfn(bovy_true, bovy_preds)
            NN   = errfn(nn_true, nn_preds)
            errs = np.array([xd, NN, spec])
            strs = ["%2.3f"%s for s in errs]
            strs[errs.argmin()] = "\\textbf{%s}"%strs[errs.argmin()]
            return strs

        # print table row
        row_strings[i] = sub(row_strings[i], 
                             div      = div_string,
                             xdmae    = format_errors(mae)[0],
                             NNmae    = format_errors(mae)[1],
                             specmae  = format_errors(mae)[2],
                             xdmape   = format_errors(mape)[0],
                             NNmape   = format_errors(mape)[1],
                             specmape = format_errors(mape)[2],
                             xdrmse   = format_errors(rmse)[0],
                             NNrmse   = format_errors(rmse)[1],
                             specrmse = format_errors(rmse)[2]
                            )

    ##############################################################################
    #### Write out to tex file
    ##############################################################################
    out_string = table_string%(row_strings[0], 
                               row_strings[1],
                               row_strings[2],
                               row_strings[3],
                               row_strings[4],
                               row_strings[5],
                               row_strings[6],
                               row_strings[7],
                               row_strings[8]
                              )
    #out_string = out_string.replace("$", "")
    with open('/Users/acm/Dropbox/Proj/astro/DESIMCMC/tex/quasar_z/nips2015/compare_table.tex', 'w') as handle:
        handle.write(out_string)



#rand_row = row_string
#rand_row = sub(rand_row, xdmae=.1)
#rand_row = sub(rand_row, xdmae=.1)
#rand_row.safe_substitute(xdmae = .1)
#rand_rowstr = rand_row.safe_substitute()

### load qso
############################################################################
### Collect GP Model and keep track of spec_ids for each test case
############################################################################
#mog_exp_file = "../analysis_mog/random/results.pkl"
#with open(mog_exp_file, 'rb') as handle:
#    rand_dict = pickle.load(handle)
#    rand_specs = rand_dict['spec_ids']
#
#mog_flux_file = "../analysis_mog/flux/results.pkl"
#with open(mog_flux_file, 'rb') as handle:
#    flux_dict = pickle.load(handle)
#    flux_specs = flux_dict['spec_ids']
#
#mog_red_file = "../analysis_mog/redshift/results.pkl"
#with open(mog_red_file, 'rb') as handle:
#    red_dict = pickle.load(handle)
#    red_specs = red_dict['spec_ids']
#
#red_res = red_dict['z_true'] - red_dict['preds']
#flux_res = flux_dict['z_true'] - flux_dict['preds']
#rand_res = rand_dict['z_true'] - rand_dict['preds']
#rand_row = sub(rand_row, 
#               specmae  = "%2.2f"%(np.mean(np.abs(rand_res))),
#               specmape = "%2.2f"%(np.mean(np.abs(rand_res/rand_dict['z_true']))),
#               specrmse = "%2.4f"%(np.sqrt(np.mean(rand_res**2))))
#
#flux_row = sub(flux_row, 
#               specmae  = "%2.2f"%(np.mean(np.abs(flux_res))),
#               specmape = "%2.2f"%(np.mean(np.abs(flux_res/flux_dict['z_true']))),
#               specrmse = "%2.4f"%(np.sqrt(np.mean(flux_res**2))))
#
#red_row = sub(red_row, 
#              specmae  = "%2.2f"%(np.mean(np.abs(red_res))),
#              specmape = "%2.2f"%(np.mean(np.abs(red_res/red_dict['z_true']))),
#              specrmse = "%2.4f"%(np.sqrt(np.mean(red_res**2))))
#
#hi_idx = rand_dict['z_true'] > 2.352
#rand_row_hi = sub(rand_row_hi,
#              specmae  = "%2.2f"%(np.mean(np.abs(red_res[hi_idx]))),
#              specmape = "%2.2f"%(np.mean(np.abs(red_res[hi_idx]/red_dict['z_true'][hi_idx]))),
#              specrmse = "%2.4f"%(np.sqrt(np.mean(red_res[hi_idx]**2))))
#
##############################################################################
#### Grab and put bovy Results in the table
##############################################################################
#bovy_exp_file = "../analysis_bovy_reproduce/bovy_exp_SPLIT-redshift_TRAIN-141591_TEST-10000_SEED-42_MAXGAUSS-20_WIDTH-0.2_UNITS-mags.pkl"
#with open(bovy_exp_file, 'rb') as handle:
#    redshift_dict = pickle.load(handle)
#    bovy_spec_ids = get_spec_ids(redshift_dict['test_idx_sub'])
#
#bovy_flux_file = "../analysis_bovy_reproduce/bovy_exp_SPLIT-flux_TRAIN-149922_TEST-10000_SEED-42_MAXGAUSS-20_WIDTH-0.2_UNITS-mags.pkl"
#with open(bovy_flux_file, 'rb') as handle:
#    flux_dict = pickle.load(handle)
#
#bovy_rand_file = "../analysis_bovy_reproduce/bovy_exp_SPLIT-random_TRAIN-83288_TEST-10000_SEED-42_MAXGAUSS-20_WIDTH-0.2_UNITS-mags.pkl"
#with open(bovy_rand_file, 'rb') as handle:
#    rand_dict = pickle.load(handle)
#    bovy_rand_ids = get_spec_ids(rand_dict['test_idx_sub'])
#
#red_res = redshift_dict['z_test'] - redshift_dict['preds_mean']
#flux_res = flux_dict['z_test'] - flux_dict['preds_mean']
#rand_res = rand_dict['z_test'] - rand_dict['preds_mean']
#
#rand_row = sub(rand_row, 
#               xdmae  = "%2.2f"%(np.mean(np.abs(rand_res))),
#               xdmape = "%2.2f"%(np.mean(np.abs(rand_res/rand_dict['z_test']))),
#               xdrmse = "%2.4f"%(np.sqrt(np.mean(rand_res**2))))
#
#flux_row = sub(flux_row, 
#               xdmae  = "%2.2f"%(np.mean(np.abs(flux_res))),
#               xdmape = "%2.2f"%(np.mean(np.abs(flux_res/flux_dict['z_test']))),
#               xdrmse = "%2.4f"%(np.sqrt(np.mean(flux_res**2))))
#
#red_row = sub(red_row, 
#              xdmae  = "%2.2f"%(np.mean(np.abs(red_res))),
#              xdmape = "%2.2f"%(np.mean(np.abs(red_res/redshift_dict['z_test']))),
#              xdrmse = "%2.4f"%(np.sqrt(np.mean(red_res**2))))
#
#print "BOVY"
#print "MAE"
#print "rand: %2.4f"%(np.mean(np.abs(rand_res)))
#print "red:  %2.4f"%(np.mean(np.abs(red_res)))
#print "flux: %2.4f"%(np.mean(np.abs(flux_res)))
#
#print "MAPE"
#print "rand: %2.4f"%(np.mean(np.abs(rand_res/rand_dict['z_test'])))
#print "red:  %2.4f"%(np.mean(np.abs(red_res/redshift_dict['z_test'])))
#print "flux: %2.4f"%(np.mean(np.abs(flux_res/flux_dict['z_test'])))
#
#print "RMSE"
#print "rand: %2.4f"%(np.sqrt(np.mean(rand_res**2)))
#print "red:  %2.4f"%(np.sqrt(np.mean(red_res**2)))
#print "flux: %2.4f"%(np.sqrt(np.mean(flux_res**2)))
#
##############################################################################
#### Neural Network 
##############################################################################
#brescia_dir = "../analysis_brescia_reproduce/"
#nn_rand_file = "brescia_exp_SPLIT-random_TRAIN-20000_TEST-all_SEED-42_NUM_LAYERS-4_WIDTH-100_UNITS-nanos.pkl"
#with open(os.path.join(brescia_dir, nn_rand_file), 'rb') as handle:
#    rand_dict = pickle.load(handle)
#
#nn_red_file = "brescia_exp_SPLIT-redshift_TRAIN-20000_TEST-all_SEED-42_NUM_LAYERS-4_WIDTH-100_UNITS-nanos.pkl"
#with open(os.path.join(brescia_dir, nn_red_file), 'rb') as handle:
#    redshift_dict = pickle.load(handle)
#
#nn_flux_file = "brescia_exp_SPLIT-flux_TRAIN-20000_TEST-all_SEED-42_NUM_LAYERS-4_WIDTH-100_UNITS-nanos.pkl"
#with open(os.path.join(brescia_dir, nn_flux_file), 'rb') as handle:
#    flux_dict = pickle.load(handle)
#
#red_res = redshift_dict['z_test'] - redshift_dict['preds'].flatten()
#flux_res = flux_dict['z_test'] - flux_dict['preds'].flatten()
#rand_res = rand_dict['z_test'] - rand_dict['preds'].flatten()
#rand_row = sub(rand_row, 
#               NNmae  = "%2.2f"%(np.mean(np.abs(rand_res))),
#               NNmape = "%2.2f"%(np.mean(np.abs(rand_res/rand_dict['z_test']))),
#               xdrmse = "%2.4f"%(np.sqrt(np.mean(rand_res**2))))
#
#flux_row = sub(flux_row, 
#               NNmae  = "%2.2f"%(np.mean(np.abs(flux_res))),
#               NNmape = "%2.2f"%(np.mean(np.abs(flux_res/flux_dict['z_test']))),
#               NNrmse = "%2.4f"%(np.sqrt(np.mean(flux_res**2))))
#
#red_row = sub(red_row, 
#              NNmae  = "%2.2f"%(np.mean(np.abs(red_res))),
#              NNmape = "%2.2f"%(np.mean(np.abs(red_res/redshift_dict['z_test']))),
#              NNrmse = "%2.4f"%(np.sqrt(np.mean(red_res**2))))
#
#print "BRESCIA"
#print "MAE"
#print "rand: %2.4f"%(np.mean(np.abs(rand_res)))
#print "red:  %2.4f"%(np.mean(np.abs(red_res)))
#print "flux: %2.4f"%(np.mean(np.abs(flux_res)))
#
#print "MAPE"
#print "rand: %2.4f"%(np.mean(np.abs(rand_res/rand_dict['z_test'])))
#print "red:  %2.4f"%(np.mean(np.abs(red_res/redshift_dict['z_test'])))
#print "flux: %2.4f"%(np.mean(np.abs(flux_res/flux_dict['z_test'])))
#
#print "RMSE"
#print "rand: %2.4f"%(np.sqrt(np.mean(rand_res**2)))
#print "red:  %2.4f"%(np.sqrt(np.mean(red_res**2)))
#print "flux: %2.4f"%(np.sqrt(np.mean(flux_res**2)))
#
#

