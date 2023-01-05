import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-hours', type=int, help='number of gpu hours to request for job', default=72)
parser.add_argument('-partition', type=str, help='Partition of gpus to access', default='dl_shared')
parser.add_argument('-allocation', type=str, help='Allocation name for billing', default='dadaist')
parser.add_argument('-env', type=str, help='Name of conda environment for running code.', default='neuromancer')
parser.add_argument('-results', type=str, help='Where to log mlflow results',
                    default='/qfs/projects/dadaist/homotopy/nlp')
parser.add_argument('-exp_folder', type=str, help='Where to save sbatch scripts and log files',
                    default='sbatch_NLP/')
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

def get_method_name(NLP_name='lincon',complexity = 10,
                   loss='penalty',
                   use_paramscheduler = 0, hyperstep=100, homo_delta=0.05,
                    obj_homo_option ='',
                   coneq_homo_option='SaS',
                   coneq_eH = 0.01, coneq_eL = 0.0,
                    conineq_homo_option='ShrinkBnds',
                    conineq_e = 0.1,
                   Q=1, Q_coneq = 50, Q_conineq = 50,
                    data_seed=408, NN_seed=408,
                    n_layers = 2, nx_hidden=30,
                    add_proj=0, proj_option='ineq',
                   ):
    # generate acopf tag
    tag = NLP_name+'C%d'%complexity
    if loss == 'augmented_lagrange':
        tag = tag + 'Lag_'
    if add_proj:
        tag = tag + 'Proj' + proj_option + '_'
    if use_paramscheduler:
        if homo_delta<0.01:
            tag = tag + 'HomoD%.3f_' % (homo_delta)
        else:
            tag = tag + 'HomoD%.2f_' % (homo_delta)
        tag = tag + ('%s_' % obj_homo_option if obj_homo_option != '' else '')
        tag = tag + ('%s_' % coneq_homo_option if coneq_homo_option != '' else '')
        if coneq_homo_option == 'SaS' and (coneq_eH != 0.01 or coneq_eL != 0.0):
            tag = tag + 'eH%.2feL%.5f_' % (coneq_eH, coneq_eL)
        tag = tag + ('%s_' % conineq_homo_option if conineq_homo_option != '' else '')
        if conineq_homo_option == 'ShrinkBnds':
            tag = tag + 'e%.2f_' % conineq_e
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
    tag = tag + 'W%d' % nx_hidden
    # constraint weight
    tag = tag + '_Qeq%d' % Q_coneq + '_Qineq%d' % Q_conineq
    return tag

NLP_name = 'lincon' #'lincon' , nlincon, NLP problem with lin or nonlin constraints
complexity = 20 #[1, 5, 10, 20, 50, 100, 200]

loss = 'penalty'
add_proj = 0
proj_option = 'ineq'

random_seed_list = [408]  # , 1, 101, 201, 301, 401]
n_layers_list = [4]
Q_coneq_list = [50]
Q_conineq_list = [50]
nx_hidden_list = [30]



"""run vanilla"""
run_vanilla = 1
if run_vanilla:
    for Q_coneq in Q_coneq_list:
        for Q_conineq in Q_conineq_list:
            for n_layers in n_layers_list:
                for nx_hidden in nx_hidden_list:
                    for seed in random_seed_list:
                        method_name = get_method_name(NLP_name='lincon',complexity = complexity,
                                       loss=loss,
                                      Q_coneq = Q_coneq, Q_conineq = Q_conineq,
                                        data_seed=seed, NN_seed=seed,
                                        n_layers = n_layers, nx_hidden=nx_hidden,
                                        add_proj=add_proj, proj_option='ineq',
                                       )
                        cmd = 'python ./mpNLP_lincon_nm_rnd_homotopy.py ' + \
                              '-gpu 0 ' + \
                              '-epochs 1000 ' + \
                              '-warmup 50 ' + \
                              '-patience 200 ' + \
                          '-complexity %s ' % complexity + \
                            '-loss %s ' % loss + \
                          '-add_proj %s ' % add_proj + \
                          '-proj_option %s ' % proj_option + \
                              '-n_layers %s ' % n_layers + \
                              '-nx_hidden %s ' % nx_hidden + \
                              '-data_seed %s ' % seed + \
                              '-NN_seed %s ' % seed + \
                              '-Q_coneq %s ' % Q_coneq + \
                              '-Q_conineq %s ' % Q_conineq + \
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
hyperstep_list = [100]
homo_delta_list = [0.1, 0.05, 0.02, 0.01, 0.005]  # [0.1, 0.05,]
obj_homo_option_list = ['OriObj', 'CvxObj']
coneq_homo_option = 'SaS'
coneq_eL_list = [0.0]  # [0.0, 0.0001]
coneq_eH_list = [0.01]  # [0.01, 0.1]
conineq_homo_option_list = ['ShrinkBnds', 'GrowPen']
conineq_e_list = [0.1]

for obj_homo_option in obj_homo_option_list:
    for coneq_eL in coneq_eL_list:
        for coneq_eH in coneq_eH_list:
            for conineq_homo_option in conineq_homo_option_list:
                for conineq_e in conineq_e_list:
                    for homo_delta in homo_delta_list:
                        for hyperstep in hyperstep_list:
                            for Q_coneq in Q_coneq_list:
                                for Q_conineq in Q_conineq_list:
                                    for n_layers in n_layers_list:
                                        for nx_hidden in nx_hidden_list:
                                            for seed in random_seed_list:
                                                method_name = get_method_name(NLP_name='lincon',complexity = complexity,
                                                           loss=loss,
                                                           use_paramscheduler = 1,
                                                            hyperstep=hyperstep, homo_delta=homo_delta,
                                                            obj_homo_option =obj_homo_option,
                                                           coneq_homo_option=coneq_homo_option,
                                                           coneq_eH = coneq_eH, coneq_eL = coneq_eL,
                                                            conineq_homo_option=conineq_homo_option,
                                                            conineq_e = conineq_e,
                                                           Q_coneq = Q_coneq, Q_conineq = Q_conineq,
                                                            data_seed=seed, NN_seed=seed,
                                                            n_layers = n_layers, nx_hidden=nx_hidden,
                                                            add_proj=add_proj, proj_option=proj_option,
                                                           )
                                                print("================================================")
                                                print(method_name)
                                                print("================================================")
                                                cmd = 'python ./mpNLP_lincon_nm_rnd_homotopy.py ' + \
                                                      '-gpu 0 ' + \
                                                      '-epochs 1000 ' + \
                                                      '-warmup 50 ' + \
                                                      '-patience 50 ' + \
                                                          '-complexity %s ' % complexity + \
                                                          '-loss %s ' % loss + \
                                                          '-add_proj %s ' % add_proj + \
                                                          '-proj_option %s ' % proj_option + \
                                                          '-n_layers %s ' % n_layers + \
                                                          '-nx_hidden %s ' % nx_hidden + \
                                                          '-data_seed %s ' % seed + \
                                                          '-NN_seed %s ' % seed + \
                                                          '-Q_coneq %s ' % Q_coneq + \
                                                          '-Q_conineq %s ' % Q_conineq + \
                                                      '-use_paramscheduler %s ' % 1 + \
                                                      '-hyperstep %s ' % hyperstep + \
                                                      '-homo_delta %s ' % homo_delta + \
                                                      '-obj_homo_option %s ' % obj_homo_option + \
                                                      '-coneq_homo_option %s ' % coneq_homo_option + \
                                                      '-coneq_eH %s ' % coneq_eH + \
                                                      '-coneq_eL %s ' % coneq_eL + \
                                                      '-conineq_homo_option %s ' % conineq_homo_option + \
                                                      '-conineq_e %s ' % conineq_e + \
                                                          '-location %s ' % args.results + \
                                                          '-logger mlflow ' + \
                                                          '-savedir %s ' % (args.results + '/'+method_name)
                                                with open(os.path.join(args.exp_folder, method_name + '.slurm'), 'w') \
                                                        as cmdfile:  # unique name for sbatch script
                                                    cmdfile.write(template + cmd)
