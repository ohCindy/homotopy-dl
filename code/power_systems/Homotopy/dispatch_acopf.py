import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='dl_shared')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='dadaist')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='neuromancer')
parser.add_argument('-results', type=str, help='Where to log mlflow results',
                    default='/qfs/projects/dadaist/homotopy/acopf')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch_opf/')
parser.add_argument('-nsamples', type=int, help='Number of samples for each experimental configuration',
                    default=1)
args = parser.parse_args()

template = '#!/bin/bash\n' +\
           '#SBATCH -A %s\n' % args.allocation +\
           '#SBATCH -t %s:00:00\n' % args.hours +\
           '#SBATCH --gres=gpu:1\n' +\
           '#SBATCH -p %s\n' % args.partition +\
           '#SBATCH -N 1\n' +\
           '#SBATCH -n 2\n' +\
           '#SBATCH -o %j.out\n' +\
           '#SBATCH -e %j.err\n' +\
           'source /etc/profile.d/modules.sh\n' +\
           'module purge\n' +\
           'module load python/anaconda3.2019.3\n' +\
           'ulimit\n' +\
           'source activate %s\n\n' % args.env

os.system('mkdir %s' % args.exp_folder)
os.system('mkdir temp')

def get_acopf_name(casename = 'case30',
                   loss='penalty',
                   use_paramscheduler = 0, hyperstep=100,
                   coneq_homo_option='SaS',
                   homo_delta=0.05, con_res_H=0.01, con_res_L=0.0,
                    add_bnds_homo=0, Pgmax_homofactor = 1,
                   Q=1, Q_coneq = 10**5, Q_conineq = 10**6,
                    data_seed=408, NN_seed=408,
                    n_layers = 2, nx_hidden=200,
                   lf_L = 0.75, lf_H = 1.5, nsamples = 50000,
                    add_pullup = 0, Q_Pgup = 1000000, tol_Pgup = 0.01,
                    add_proj=0, proj_option='ineq',
                   ):
    tag = casename
    if loss == 'augmented_lagrange':
        tag = tag + 'Lag_'
    if add_proj:
        tag = tag + 'Proj' + proj_option + '_'
    if use_paramscheduler:
        if homo_delta < 0.01:
            tag = tag + '%sDelta%.3f_' % (coneq_homo_option, homo_delta)
        else:
            tag = tag + '%sDelta%.2f_'%(coneq_homo_option, homo_delta)
        if coneq_homo_option == 'SaS' and (con_res_H!=0.1 or con_res_L!=0.0001):
            tag = tag + 'H%.2fL%.5f'%(con_res_H, con_res_L)
        if add_bnds_homo:
            tag = tag + 'HomoBnds'
            if Pgmax_homofactor!=1:
                tag = tag + 'Pg%d'%Pgmax_homofactor
            tag = tag + '_'
    else:
        tag = tag + 'vanilla_'

    if use_paramscheduler and hyperstep != 100:
        tag = tag + 'OuterStep%d_' % hyperstep
    #
    if Q != 1:
        tag = tag + 'Q%.1f' % Q
    #
    if data_seed != 408 or NN_seed != 408:
        tag = tag + 'dseed%dnnseed%d_' % (data_seed, NN_seed)
    tag = tag + 'L%d' % n_layers
    tag = tag + ('W%d' % nx_hidden if nx_hidden != 200 else '')
    #
    tag = tag + '_%dsamples' % nsamples
    if lf_L != 0.75 or lf_H != 1.5:
        tag = tag + 'lfL%.2fH%.2f' % (lf_L, lf_H)
    #
    # PUll up Pg and Vg
    if add_pullup and (Q_Pgup > 0):
        # PgUp, VgUp, or PgVgUp
        tag = tag + '_' + ('Pg' if Q_Pgup > 0 else '') + (
            'tol%.2f' % tol_Pgup if tol_Pgup > 0 else '') + 'Up'
    # equality constraint weight
    if Q_coneq != 1e5:
        tag = tag + '_Qeq%d' % Q_coneq
    if Q_conineq != 1e6:
        tag = tag + '_Qineq%d' % Q_conineq
    return tag

casename = 'case30'

loss = 'penalty'
add_proj = 0
proj_option = 'ineq'
#
random_seed_list = [408]  # , 1, 101, 201, 301, 401]
n_layers_list = [2]
nx_hidden_list = [200]
Q_coneq_list = [1e5]
Q_conineq_list = [1e6]
#
add_pullup_list = [0, 1]
tol_Pgup = 0.01

"""run vanilla"""
run_vanilla = 1
if run_vanilla:
    for add_pullup in add_pullup_list:
        for Q_coneq in Q_coneq_list:
            for Q_conineq in Q_conineq_list:
                for n_layers in n_layers_list:
                    for nx_hidden in nx_hidden_list:
                        for seed in random_seed_list:
                            method_name = get_acopf_name(casename = casename, loss = loss,
                                                         add_proj=add_proj, proj_option=proj_option,
                                                         n_layers=n_layers, nx_hidden = nx_hidden,
                                                         data_seed=seed, NN_seed=seed,
                                                         Q_coneq=Q_coneq, Q_conineq=Q_conineq,
                                                         #
                                                         add_pullup=add_pullup,tol_Pgup=tol_Pgup
                                                         )
                            cmd = 'python ./mpACOPF_nm_homotopy.py ' + \
                                  '-gpu 0 ' + \
                                  '-epochs 1000 ' + \
                                  '-warmup 50 ' + \
                                  '-patience 200 ' + \
                              '-casename %s ' % casename + \
                                  '-loss %s ' % loss + \
                                  '-add_proj %s ' % add_proj + \
                              '-proj_option %s ' % proj_option + \
                                  '-n_layers %s ' % n_layers + \
                                  '-nx_hidden %s ' % nx_hidden + \
                                  '-data_seed %s ' % seed + \
                                  '-NN_seed %s ' % seed + \
                                  '-Q_coneq %s ' % Q_coneq + \
                                  '-Q_conineq %s ' % Q_conineq + \
                              '-add_pullup %s ' % add_pullup + \
                              '-tol_Pgup %s ' % tol_Pgup + \
                                  '-location %s ' % args.results + \
                                  '-logger mlflow ' + \
                                  '-savedir %s ' % (args.results + '/'+method_name)
                                  # '-savedir %s/vanilla_L%s_Q%s_%s_seed%s ' \
                                  # % (casename, n_layers, Q_coneq, Q_conineq, seed)
                            print("================================================")
                            print(method_name)
                            print("================================================")
                            with open(os.path.join(args.exp_folder, method_name+'.slurm') , 'w')\
                                    as cmdfile:  # unique name for sbatch script
                                cmdfile.write(template + cmd)

coneq_homo_option_list = ['SaS', 'Tx-stepping', 'Load-stepping', ]
hyperstep_list = [100]
homo_delta_list = [0.1, 0.05, 0.02, 0.01, 0.005] #[0.1, 0.05,]
bnds_homo_list = [1, 0]
#
con_res_L = 0.0
con_res_H = 0.01
# update
add_pullup_list = [1]

for add_bnds_homo in bnds_homo_list:
    for coneq_homo_option in coneq_homo_option_list:
        for homo_delta in homo_delta_list:
            for hyperstep in hyperstep_list:
                for add_pullup in add_pullup_list:
                    for Q_coneq in Q_coneq_list:
                        for Q_conineq in Q_conineq_list:
                            for n_layers in n_layers_list:
                                for nx_hidden in nx_hidden_list:
                                    for seed in random_seed_list:
                                        method_name = get_acopf_name(casename=casename,
                                                                     loss=loss,
                                             add_proj=add_proj, proj_option=proj_option,
                                             n_layers=n_layers, nx_hidden=nx_hidden,
                                             data_seed=seed, NN_seed=seed,
                                             Q_coneq=Q_coneq, Q_conineq=Q_conineq,
                                             #
                                             add_pullup=add_pullup, tol_Pgup=tol_Pgup,
                                            #
                                            use_paramscheduler = 1, hyperstep = hyperstep,
                                            coneq_homo_option = coneq_homo_option,
                                            homo_delta = homo_delta, add_bnds_homo=add_bnds_homo,
                                            con_res_L = con_res_L, con_res_H=con_res_H,
                                             )
                                        print("================================================")
                                        print(method_name)
                                        print("================================================")
                                        cmd = 'python ./mpACOPF_nm_homotopy.py ' + \
                                              '-gpu 0 ' + \
                                              '-epochs 1000 ' + \
                                              '-warmup 50 ' + \
                                              '-patience 50 ' + \
                                                  '-casename %s ' % casename + \
                                                '-loss %s ' % loss + \
                                                '-add_proj %s ' % add_proj + \
                                                  '-proj_option %s ' % proj_option + \
                                                  '-n_layers %s ' % n_layers + \
                                                  '-nx_hidden %s ' % nx_hidden + \
                                                  '-data_seed %s ' % seed + \
                                                  '-NN_seed %s ' % seed + \
                                                  '-Q_coneq %s ' % Q_coneq + \
                                                  '-Q_conineq %s ' % Q_conineq + \
                                                      '-add_pullup %s ' % add_pullup + \
                                                      '-tol_Pgup %s ' % tol_Pgup + \
                                              '-use_paramscheduler %s ' % 1 + \
                                              '-hyperstep %s ' % hyperstep + \
                                              '-coneq_homo_option %s ' % coneq_homo_option + \
                                              '-homo_delta %s ' % homo_delta + \
                                              '-add_bnds_homo %s ' % add_bnds_homo + \
                                              '-con_res_L %s ' % con_res_L + \
                                              '-con_res_H %s ' % con_res_H + \
                                                  '-location %s ' % args.results + \
                                                  '-logger mlflow ' + \
                                                  '-savedir %s ' % (args.results + '/'+method_name)
                                        with open(os.path.join(args.exp_folder, method_name + '.slurm'), 'w') \
                                                as cmdfile:  # unique name for sbatch script
                                            cmdfile.write(template + cmd)
