"""
meta learning:
try running models with different hyper parameter settings
"""

import pypower
import numpy as np
import torch

import neuromancer.arg as arg
from mpACOPF_nm_homotopy import main as run_model
import copy
from os.path import exists

import nm_extentions_for_opf

"""
---meta learning class
"""
class ModelSet:
    def __init__(self, run_func, args_dict = dict()):
        self.args_dict = args_dict
        self.run_func = run_func # run_func(args)
        self.out_dicts = dict()
    def run_models(self, model_keys=None):
        if not model_keys:
            model_keys = self.args_dict.keys()
        for model_key in model_keys:
            print("===================================================")
            print("||                    Runing %s                   ||"%model_key)
            print("===================================================")
            out_dict = self.run_func(self.args_dict[model_key])
            self.out_dicts[model_key] = out_dict
    def overview_models(self, model_keys = None):
        if not model_keys:
            model_keys = self.args_dict.keys()
        for model_key in model_keys:
            print("====================================")
            print(model_key)
            print(self.out_dicts[model_key]['performance'])



def arg_mpACOPF_problem(prefix=''):
    """
    Command line parser for mpED problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpACOPF")
    #
    gp.add("-is_train", type=int, default=0,  #
           help="whether to train a new model or load existing models and print.")
    # Homotopy and hyper tuning
    gp.add("-use_paramscheduler", type=int, default=0,
           help="choices: 0, 1; whether or not use homotopy")
    gp.add("-hyperstep", type=int, default=100,
           help="adjust hyper parameters like Q, every ? epochs")
    gp.add("-homo_delta", type=float, default=0.05,  # 0.05,
           help="d_lambdaH: update stepsize of homotopy parameter ")
    gp.add("-coneq_homo_option", type=str, default='SaS',  #
           help="what eq constraint homotopy: Tx-stepping, Load-stepping, SaS (split-and-shrink)")
    gp.add("-con_res_L", type=float, default=0.0,
           help="SaS: PQ balance <= (1-lambda)*res_H+res_L,PQ balance >= -(1-lambda)*res_H-res_L")
    gp.add("-con_res_H", type=float, default=0.01,
           help="SaS: PQ balance <= (1-lambda)*res_H+res_L,PQ balance >= -(1-lambda)*res_H-res_L")
    gp.add("-add_bnds_homo", type=int, default=0,  #
           help="choices: 0, 1: homotopy on ineq constraints: True: use ShinkBnds, False: use GrowPenalty")
    gp.add("-Pgmax_homofactor", type=float, default=1,  #
           help="Pgmax_homo = Pgmax_ori + (1-lambda_H)*Pgmax_homofactor")
    gp.add("-Q_warmhomo", type=float, default=1000000,
           help="weight of homotopy warm-up homo loss Q||x-x_trivial||, for load stepping")
    #
    # domain knowledge: avoid underestimate of Pg and |Vg|
    gp.add("-add_pullup", type=int, default = 0,  #
           help="choices: 0, 1: whether or not to drag Pg and |Vg| up via loss and constraints.")
    gp.add("-Q_Pgup", type=float, default=1000000,
           help="weight of constraint total Pg>= total Pd, to pull Pg up")
    gp.add("-tol_Pgup", type=float, default=0.01,
           help="weight of constraint total Pg>= total Pd + tol(pu), to pull Pg up")
    # correction layer based on projected gradient update
    gp.add("-add_proj", type=int, default=0,
           help="choices: 0, 1: Whether to use projected gradient update for correction of the output variables")
    gp.add("-proj_option", type=str, default='ineq',
           help="correct all / eq / ineq constraints by projected gradient update")
    # other weights and training settings
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_coneq", type=float, default=1e5,  # 100000.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-Q_conineq", type=float, default=1e6,  # 1000000.0,
           help="constraints penalty weight.")  # tuned value
    gp.add("-nx_hidden", type=int, default=200,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-NN_seed", type=int, default=408,
           help="random seed for pytorch NN")
    gp.add("-epochs", type=int, default=1000,  # 200
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.0001,  # 0.01
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=50,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=50,  # 100
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-casename", type=str, default="case30",
           help="Network file name")
    gp.add("-nsamples", type=int, default=50000,
           help="Number of samples")
    gp.add("-lf_L", type=float, default=0.75,
           help="randomly samples total Pd within [lf_L, lf_H]*Pdbase")
    gp.add("-lf_H", type=float, default=1.5,
           help="randomly samples total Pd within [lf_L, lf_H]*Pdbase")
    gp.add("-loss", type=str, default='penalty',  # 'penalty',
           choices=['penalty', 'augmented_lagrange', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-eta", type=float, default=0.99,
           help="eta in augmented lagrangian.")
    gp.add("-sigma", type=float, default=2.0,
           help="sigma in augmented lagrangian.")
    gp.add("-mu_init", type=float, default=1.,
           help="mu_init in augmented lagrangian.")
    gp.add("-mu_max", type=float, default=1000.,
           help="mu_max in augmented lagrangian.")
    gp.add("-inner_loop", type=int, default=1,
           help="inner loop in augmented lagrangian")

    return parser


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """

    """===prepare base args0==="""
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpACOPF_problem()])
    args0, grps = parser.parse_arg_groups()
    args0.bias = True
    device = f"cuda:{args0.gpu}" if args0.gpu is not None else "cpu"
    """
    # prepare a distribution of args
    """
    args_dict = dict()
    meta_name = 'test_mpACOPF_homotopy' #args0.savedir
    args0.savedir = meta_name
    args0.casename = 'case30'
    args0.is_train = 0

    # args0.nsamples = 50000
    # args0.lf_H = 1.25
    # args0.nx_hidden = 400
    # args0.loss = 'augmented_lagrange'
    args0.add_proj = 0
    args0.proj_option = 'ineq'
    """creat different args to run"""

    data_seed_list = [408] #, 1, 101, 201, 301, 401]
    NN_seed_list = [408]
    n_layers_list = [2]
    Q_coneq_list = [1e5]
    Q_conineq_list = [1e6]


    n_model = 0
    """1. vanilla"""
    #
    for Q_coneq in Q_coneq_list:
        for Q_conineq in Q_conineq_list:
            for n_layers in n_layers_list:
                for NN_seed in NN_seed_list:
                    for data_seed in data_seed_list:
                        this_args = copy.deepcopy(args0)
                        #
                        this_args.patience = 200
                        this_args.data_seed = data_seed
                        this_args.NN_seed = NN_seed
                        this_args.n_layers = n_layers
                        this_args.Q_coneq = Q_coneq
                        this_args.Q_conineq = Q_conineq
                        #
                        tag = nm_extentions_for_opf.get_tag_acopf(this_args)
                        model_key = '/%s/%s' % (this_args.casename, tag)
                        this_args.savedir = meta_name + model_key
                        #
                        args_dict[model_key] = this_args
                        n_model = n_model + 1

    """2. vanilla pull up"""
    tol_Pgup_list = [0.01] #
    for tol_Pgup in tol_Pgup_list:
        for Q_coneq in Q_coneq_list:
            for Q_conineq in Q_conineq_list:
                for n_layers in n_layers_list:
                    for NN_seed in NN_seed_list:
                        for data_seed in data_seed_list:
                            this_args = copy.deepcopy(args0)
                            #
                            this_args.patience = 200
                            this_args.add_pullup = 1
                            this_args.tol_Pgup = tol_Pgup
                            #
                            this_args.data_seed = data_seed
                            this_args.NN_seed = NN_seed
                            this_args.n_layers = n_layers
                            this_args.Q_coneq = Q_coneq
                            this_args.Q_conineq = Q_conineq
                            #
                            tag = nm_extentions_for_opf.get_tag_acopf(this_args)
                            model_key = '/%s/%s' % (this_args.casename, tag)
                            this_args.savedir = meta_name + model_key
                            #
                            args_dict[model_key] = this_args
                            n_model = n_model + 1

    """constraint homotopy with Pg pull up"""
    coneq_homo_option_list = ['Tx-stepping','Load-stepping']
    hyperstep_list = [100]
    homo_delta_list = [0.05,] #[0.1, 0.05,]
    bnds_homo_list = [1, 0]
    for add_bnds_homo in bnds_homo_list:
        for coneq_homo_option in coneq_homo_option_list:
            for homo_delta in homo_delta_list:
                for tol_Pgup in tol_Pgup_list:
                    for hyperstep in hyperstep_list:
                        for Q_coneq in Q_coneq_list:
                            for Q_conineq in Q_conineq_list:
                                for n_layers in n_layers_list:
                                    for NN_seed in NN_seed_list:
                                        for data_seed in data_seed_list:
                                            this_args = copy.deepcopy(args0)
                                            #
                                            this_args.add_pullup = 1
                                            this_args.tol_Pgup = tol_Pgup
                                            #
                                            this_args.use_paramscheduler = 1
                                            this_args.hyperstep = hyperstep
                                            this_args.coneq_homo_option = coneq_homo_option
                                            this_args.homo_delta = homo_delta
                                            this_args.add_bnds_homo = add_bnds_homo
                                            #
                                            this_args.data_seed = data_seed
                                            this_args.NN_seed = NN_seed
                                            this_args.n_layers = n_layers
                                            this_args.Q_coneq = Q_coneq
                                            this_args.Q_conineq = Q_conineq
                                            #
                                            tag = nm_extentions_for_opf.get_tag_acopf(this_args)
                                            model_key = '/%s/%s' % (this_args.casename, tag)
                                            this_args.savedir = meta_name + model_key
                                            #
                                            args_dict[model_key] = this_args
                                            n_model = n_model + 1

    # general homotopy SaS (split and shrink)
    con_res_L_list = [0.0] #[0.0, 0.0001]
    con_res_H_list = [0.01] #[0.01, 0.1]
    homo_delta_list = [0.05]
    bnds_homo_list = [1, 0]
    for con_res_L in con_res_L_list:
        for con_res_H in con_res_H_list:
            for add_bnds_homo in bnds_homo_list:
                for homo_delta in homo_delta_list:
                    for tol_Pgup in tol_Pgup_list:
                        for hyperstep in hyperstep_list:
                            for Q_coneq in Q_coneq_list:
                                for Q_conineq in Q_conineq_list:
                                    for n_layers in n_layers_list:
                                        for NN_seed in NN_seed_list:
                                            for data_seed in data_seed_list:
                                                this_args = copy.deepcopy(args0)
                                                #
                                                this_args.add_pullup = 1
                                                this_args.tol_Pgup = tol_Pgup
                                                #
                                                this_args.use_paramscheduler = 1
                                                this_args.hyperstep = hyperstep
                                                this_args.coneq_homo_option = 'SaS'
                                                this_args.con_res_L = con_res_L
                                                this_args.con_res_H = con_res_H
                                                this_args.homo_delta = homo_delta
                                                this_args.add_bnds_homo = add_bnds_homo
                                                #
                                                this_args.data_seed = data_seed
                                                this_args.NN_seed = NN_seed
                                                this_args.n_layers = n_layers
                                                this_args.Q_coneq = Q_coneq
                                                this_args.Q_conineq = Q_conineq
                                                #
                                                tag = nm_extentions_for_opf.get_tag_acopf(this_args)
                                                model_key = '/%s/%s' % (this_args.casename, tag)
                                                this_args.savedir = meta_name + model_key
                                                #
                                                args_dict[model_key] = this_args
                                                n_model = n_model + 1

    nm_set = ModelSet(run_func=run_model, args_dict=args_dict)
    print("models to run:")
    for i,key in enumerate(nm_set.args_dict.keys()):
        print(i, key)
    print("stored under meta dir", meta_name)
    nm_set.run_models(nm_set.args_dict.keys())
    nm_set.overview_models()
    breakpoint = 1
