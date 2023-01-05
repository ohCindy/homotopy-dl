"""
meta learning:
try running models with different hyper parameter settings
"""

import pypower
import numpy as np
import torch

import neuromancer.arg as arg
from mpNLP_lincon_nm_rnd_homotopy import main as run_model
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


def arg_mpNLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpMIMLP")
    #
    gp.add("-is_train", type=int, default=0,  #
           help="0: False, 1: True; whether to train a new model or load existing models and print.")
    # homotopy settings
    gp.add("-use_paramscheduler", type=int, default=0,
           help="choices: 0, 1: whether or not use homotopy, or other hyperpara tuning in outerloops")
    gp.add("-hyperstep", type=int, default=100,
           help="adjust hyper parameters, every ? epochs")
    gp.add("-homo_delta", type=float, default= 0.05, #0.05,
           help="update step of homotopy parameter ")
    gp.add("-obj_homo_option", type=str, default='OriObj',  #
           help="choices: 1)CvxObj: fobj = lambdaH*fobj_ori + (1-lambdaH)f_cvx 2)Oriobj : no change")
    gp.add("-coneq_homo_option", type=str, default='SaS',  #
           help="what eq constraint homotopy: Tx-stepping, Load-stepping, SaS (split-and-shrink)")
    gp.add("-coneq_eL", type = float, default = 0.0,
           help = "SaS: h <= (1-lambda)*res_H+res_L, h >= -(1-lambda)*eH-eL")
    gp.add("-coneq_eH", type=float, default=0.01,
           help="SaS: h <= (1-lambda)*res_H+res_L, h >= -(1-lambda)*eH-eL")
    gp.add("-conineq_homo_option", type=str, default='ShrinkBnds',  #
           help="homotopy heuristic option for ineq constraints: ShrinkBnds, GrowPen")
    gp.add("-conineq_e", type=float, default=0.1,
           help="shrink bound: g<=(1-lambda)*conineq_e")
    # correction layer based on projected gradient update
    gp.add("-add_proj", type=int, default=0,
           help="choices: 0, 1: Whether to use projected gradient update for correction of the output variables")
    gp.add("-proj_option", type = str, default = 'ineq',
           help="correct all / eq / ineq constraints by projected gradient update")
    # problem settings
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_coneq", type=float, default=50.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-Q_conineq", type=float, default=50.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-nx_hidden", type=int, default=30,
           help="Number of hidden states of the solution map per problem dimension")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-NN_seed", type=int, default=408,
           help="random seed for pytorch NN")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-loss", type=str, default='penalty',
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
    gp.add("-complexity", type=int, default=5,
           help="number of decision variables and constraints")
    return parser

if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """

    """===prepare base args0==="""
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpNLP_problem()])
    args0, grps = parser.parse_arg_groups()
    args0.bias = True
    device = f"cuda:{args0.gpu}" if args0.gpu is not None else "cpu"
    """
    # prepare a distribution of args
    """
    args_dict = dict()
    meta_name = 'test_mpNLP_lincon_rnd_homotopy' #args0.savedir
    args0.savedir = meta_name
    args0.complexity = 50
    args0.is_train = 0


    # args0.loss = 'augmented_lagrange'
    args0.add_proj = 0
    """creat different args to run"""

    data_seed_list = [408] #, 1, 101, 201, 301, 401]
    NN_seed_list = [408]
    n_layers_list = [2]
    Q_coneq_list = [50]
    Q_conineq_list = [50]


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
                        tag = nm_extentions_for_opf.get_tag_general(this_args)
                        model_key = '/complexity%s/%s' % (this_args.complexity, tag)
                        this_args.savedir = meta_name + model_key
                        #
                        args_dict[model_key] = this_args
                        n_model = n_model + 1


    """constraint homotopy SaS"""
    hyperstep_list = [100]
    homo_delta_list = [0.1] #[0.1, 0.05,]
    obj_homo_option_list = ['OriObj']
    coneq_homo_option = 'SaS'
    coneq_eL_list = [0.0] #[0.0, 0.0001]
    coneq_eH_list = [0.01] #[0.01, 0.1]
    conineq_homo_option_list = ['ShrinkBnds', 'GrowPen']
    conineq_e_list = [0.1, 0.01]
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
                                            for NN_seed in NN_seed_list:
                                                for data_seed in data_seed_list:
                                                    this_args = copy.deepcopy(args0)
                                                    #
                                                    this_args.patience = 50
                                                    #
                                                    this_args.use_paramscheduler = True
                                                    this_args.hyperstep = hyperstep
                                                    this_args.obj_homo_option = obj_homo_option
                                                    this_args.coneq_homo_option = coneq_homo_option
                                                    this_args.coneq_eL = coneq_eL
                                                    this_args.coneq_eH = coneq_eH
                                                    this_args.homo_delta = homo_delta
                                                    this_args.conineq_homo_option = conineq_homo_option
                                                    this_args.conineq_e = conineq_e
                                                    #
                                                    this_args.data_seed = data_seed
                                                    this_args.NN_seed = NN_seed
                                                    this_args.n_layers = n_layers
                                                    this_args.Q_coneq = Q_coneq
                                                    this_args.Q_conineq = Q_conineq
                                                    #
                                                    tag = nm_extentions_for_opf.get_tag_general(this_args)
                                                    model_key = '/complexity%s/%s' % (this_args.complexity, tag)
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
