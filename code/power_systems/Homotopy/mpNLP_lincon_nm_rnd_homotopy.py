"""
multi-parametric nonlinear program (mpNLP)
using Neuromancer's  differentiable parametric programming (DPP) approach

Objective: multi-dimensional rosenbrock
    see: https://en.wikipedia.org/wiki/Test_functions_for_optimization

subject to: random linear inequalities

"""


import numpy as np
import torch
from torch.utils.data import DataLoader
import slim
import cvxpy as cp
from cvxpy import *

#from neuromancer.trainer import Trainer
from trainer_new import Trainer
import trainer_new
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Loss
from neuromancer.activations import activations
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import get_static_dataloaders
from neuromancer.loss import get_loss
from neuromancer.maps import Map
from neuromancer import blocks
from neuromancer.integers import IntegerProjection, IntegerInequalityProjection

# correction
from neuromancer.solvers import GradientProjection
import nm_extentions_for_opf

# homotopy
from nm_extentions_for_opf import param_scheduler

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


def main(args):
    """
    # # #  optimization problem hyperparameters
    """
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    """
    # # #  Problem setup
    """
    # Parameters to change
    # n_x = args.complexity                       # number of decision variables
    # n_theta = int(np.ceil(args.complexity/2))          # number of parameters
    # n_con = args.complexity                     # number of constraints

    n_x = 5 * args.complexity  # number of decision variables
    n_theta = 2 * args.complexity  # number of parameters
    n_con = 5 * args.complexity  # number of

    args.nx_hidden = args.nx_hidden * int(np.sqrt(n_x))

    # fixed problem parameters
    A = torch.randn(n_x, n_con).t().to(device)
    b = torch.randn(n_con).to(device)
    E = torch.randn(n_theta, n_con).t().to(device)

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta
    nsim = 50000  # number of datapoints: increase sample density for more robust results
    samples = {"theta": 0.5 * np.random.randn(nsim, n_theta)}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpMINLP problem formulation in Neuromancer
    """
    func = blocks.MLP(insize=n_theta, outsize=n_x,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[args.nx_hidden] * args.n_layers)
    torch.manual_seed(args.NN_seed)
    nm_extentions_for_opf.NN_initializer(func, method='xavier')
    sol_map = Map(func,
            input_keys=["theta"],
            output_keys=["x"],
            name='primal_map')

    # variables
    x = Variable('x')  # decision variable as output from the solution map
    x_i = Variable('x')[:, :-1]
    x_j = Variable('x')[:, 1:]
    theta = Variable('theta')  # sampled parametric variable

    # objective function
    f = (1-x_i)**2 + 2*(x_j-x_i**2)**2
    obj = f.minimize(weight=args.Q, name='obj')

    # constraints
    con_I = args.Q_conineq * (x@A.t() <= b + theta@E.t())


    """parameter tuning outerloops and homotopy"""
    Qstep = args.hyperstep if args.use_paramscheduler else args.epochs
    if args.use_paramscheduler:
        homo_param_sche = param_scheduler(Q0=0, stepsize=Qstep, mode='add', factor=args.homo_delta, Qmax=1,
                                          name='homotopy-parameter-(%s)' % args.coneq_homo_option)
    else:  # not necessarily needed just use it to clearly separate
        homo_param_sche = param_scheduler(Q0=1, stepsize=Qstep, factor=1, name='homotopy-parameter')

    MinOuterLoops = [args.epochs // Qstep]
    if args.use_paramscheduler:
        MinOuterLoops.append(np.ceil(1.0 / args.homo_delta) + 5)
    MinOuterLoops = int(np.max(MinOuterLoops))
    #
    nm_sol_rec = []
    metric_rec = []

    best_res = 100.0

    for current_epoch in range(MinOuterLoops):
        print("--------OuterLoop # %d -----------" % current_epoch)
        """ adjust loss and loss weights based on homotopy"""
        #
        if current_epoch > 0 and current_epoch >= MinOuterLoops - 2:
            args.lr = args.lr * 0.1
        # update homotopy parameter
        lambda_H = homo_param_sche.step(current_epoch * Qstep)

        """update constraints based on homotopy settings"""
        if args.use_paramscheduler:
            #obj
            if args.obj_homo_option == 'CvxObj':
                f_homo = (1 - x_i) ** 2 + \
                         lambda_H * 2 * (x_j - x_i ** 2) ** 2 +\
                         (1-lambda_H) * (x_j**2 + x_i**4)
                obj = f_homo.minimize(weight=args.Q, name='obj')
            #eq con
            if args.coneq_homo_option == 'SaS':
                con_E_left = None
                con_E_right = None
            #ineq con
            if args.conineq_homo_option=='ShrinkBnds':
                con_I = args.Q_conineq * (x@A.t() <= b + theta@E.t() + (1-lambda_H)*args.conineq_e)
            elif args.conineq_homo_option == 'GrowPen':
                con_I.weight = args.Q_conineq*lambda_H

        # list of objectives, constraints, and components of solution map
        objectives = [obj]
        constraints = [con_I]
        components = [sol_map]

        #
        if args.add_proj:  # use projected gradient update
            project_keys = ["x"]
            if args.proj_option == "ineq":
                cons_proj = [con_I]
            else:
                print("nothing for projection")
                raise "no projection configuration"
            projection = GradientProjection(cons_proj, input_keys=project_keys,
                                            num_steps=5, name='proj_correction')
            components.append(projection)
            print("Projection layer added to the components")

        # create constrained optimization loss
        loss = get_loss(objectives, constraints, train_data, args)
        # construct constrained optimization problem
        problem = Problem(components, loss, grad_inference=bool(args.add_proj))
        problem = problem.to(device)

        """
        # # # Metrics and Logger
        """
        savedir = args.savedir + '/Outerloop%d' % current_epoch if args.use_paramscheduler else args.savedir
        args.verbosity = 1
        metrics = ["dev_loss"]
        logger = BasicLogger(args=args, savedir=savedir, verbosity=args.verbosity, stdout=metrics)
        logger.args.system = 'mpNLP_lincon'

        """
        # # #  mpNLP problem solution in Neuromancer
        """
        optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

        # define trainer
        trainer = Trainer(
            problem,
            train_data,
            dev_data,
            test_data,
            optimizer,
            logger=logger,
            epochs=Qstep,  # args.epochs,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric="dev_loss",
            patience=args.patience,
            warmup=args.warmup,
            device=device,
            lr_scheduler=True if (not args.use_paramscheduler) else False
        )


        # Train mpLP solution map, or load existing model
        trainer.current_epoch = current_epoch * Qstep
        is_train = args.is_train
        if not is_train:
            try:
                model_dir = savedir + '/best_model_state_dict.pth'
                print("attempting to load model from ", model_dir)
                best_model = torch.load(model_dir)
                problem.load_state_dict(best_model)
            except FileNotFoundError:
                print("model in this outerloop not found, train one, still attempt to load model in the next outerloop")
                is_train = 1
        if is_train:
            best_model = trainer.train()

        best_outputs = trainer.test(best_model)

        """
        Performance evaluation
        """
        # fixed problem parameters

        # get test data
        ntest = 100
        datapoint = test_data.dataset.get_full_batch()
        datapoint['theta'] = datapoint['theta'][0:ntest]
        datapoint['name'] = "test"

        model_out = problem(trainer_new.move_batch_to_device(datapoint, device=device))
        x_nm_test = model_out['test_' + "x"].detach().view(ntest, n_x)
        x_i_test = x_nm_test[:, :-1]
        x_j_test = x_nm_test[:, 1:]
        theta_test =  datapoint['theta'].to(device)
        # objective function
        obj_nm_test = (1 - x_i_test) ** 2 + 2 * (x_j_test - x_i_test ** 2) ** 2
        obj_nm_test = obj_nm_test.sum(dim=-1) #size (ntest)
        obj_nm_test = obj_nm_test.mean().to("cpu")

        res_ineq_viol = torch.maximum(torch.zeros(1).to(device), x_nm_test @ A.t() - b - theta_test @ E.t())
        res_ineq_viol = res_ineq_viol.to("cpu")
        ave_ineq_viol = res_ineq_viol.mean(dim=-1)
        ave_ineq_viol_mean = ave_ineq_viol.mean().detach().numpy().item()
        ave_ineq_viol_std = ave_ineq_viol.std().detach().numpy().item()
        max_ineq_viol, _ = res_ineq_viol.max(dim=-1)
        max_ineq_viol_mean = max_ineq_viol.mean().detach().numpy().item()
        max_ineq_viol_std = max_ineq_viol.std().detach().numpy().item()
        #
        print(f'Neuromancer obj ', obj_nm_test)
        print(f'ineq viol mean %.4f (%.4f) max %.4f (%.4f)'%(
            ave_ineq_viol_mean, ave_ineq_viol_std, max_ineq_viol_mean, max_ineq_viol_std
        ))
        # save metrics
        this_metric = [obj_nm_test,
                       ave_ineq_viol_mean, ave_ineq_viol_std,
                       max_ineq_viol_mean, max_ineq_viol_std]
        metric_rec.append(this_metric)
        # update best metric
        if max_ineq_viol_mean < best_res:
            best_res = max_ineq_viol_mean
            #
            best_obj = obj_nm_test
            best_ave_ineq_viol_mean, best_ave_ineq_viol_std = ave_ineq_viol_mean, ave_ineq_viol_std
            best_max_ineq_viol_mean, best_max_ineq_viol_std = max_ineq_viol_mean, max_ineq_viol_std

    metric_names = ['mean obj_nm',
                    'ineq (mean viol)', 'std', '(max viol)', 'std',
                    ]
    metric_formats = ['%.2f',
                      '%.4f', '%.4f', '%.4f', '%.4f',
                     ]
    for istep in range(len(nm_sol_rec)):
        print("Outer:", istep,
              [('%s ' + metric_formats[imetric]) % (metric_names[imetric], val) for imetric, val in
               enumerate(metric_rec[istep])],
              )
    #
    title = ('obj %.2f, ineq viol mean %.4f(%.4f),max %.4f(%.4f)') \
            % (best_obj,
               best_ave_ineq_viol_mean, best_ave_ineq_viol_std,
               best_max_ineq_viol_mean, best_max_ineq_viol_std
               )
    out_dict = dict()
    out_dict['performance'] = title + '\n'

    with open(args.savedir + '/result.txt', 'w') as f:
        f.write(args.savedir + '\n')
        f.write(out_dict['performance'])
        f.write('\n')
    print(args.savedir + '/result.txt')
    print(out_dict['performance'])

    return out_dict

if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpNLP_problem()])
    args, grps = parser.parse_arg_groups()
    print("add_proj in __main__", args.add_proj)
    #
    tag = nm_extentions_for_opf.get_tag_general(args)
    print("validate tag: ", tag)
    print("add_proj in __main__", args.add_proj)
    if args.savedir == "test": #"test" is the built-in default savedir
        args.savedir = "test_mpNLP_lincon_rnd_homotopy" + '/complexity%s/%s' % (args.complexity, tag)
    print("model save directory", args.savedir)

    #
    out_dict = main(args)
    import os
    from datetime import date
    today = date.today()
    with open(os.path.dirname(args.savedir) + '/result_nlp.txt', 'a') as f: #with open(args.savedir + '/result.txt', 'w') as f:
        f.write('\n')
        f.write(str(today) + ' ' + args.savedir + '\n')
        f.write(tag + '\n')
        f.write(out_dict['performance'])

    print(os.path.dirname(args.savedir) + '/result_opf.txt')

    mark_end = 1