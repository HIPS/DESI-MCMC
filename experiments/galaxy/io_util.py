import os
from sample_stamp import load_samples

def load_mcmc_chains(chain_file_0, num_chains=4, burnin=2500):
    src_samp_chains = []
    ll_samp_chains  = []
    eps_samp_chains = []
    for i in range(num_chains):
        chain_i_file = chain_file_0.replace("chain_0", "chain_%d"%i)
        if not os.path.exists(chain_i_file):
            print "chain_file: %s does not exist"%chain_file_0
            continue
        samp_dict = load_samples(chain_i_file)
        if samp_dict['srcs'][burnin] == samp_dict['srcs'][burnin+1]:
            print "chain_file: %s has zeros?"%chain_file_0
            print samp_dict['srcs'][burnin:(burnin+2)]
            continue

        src_samp_chains.append(samp_dict['srcs'][burnin:,0])
        ll_samp_chains.append(samp_dict['ll'][burnin:])
        eps_samp_chains.append(samp_dict['epsilon'][burnin:])
    return src_samp_chains, ll_samp_chains, eps_samp_chains


