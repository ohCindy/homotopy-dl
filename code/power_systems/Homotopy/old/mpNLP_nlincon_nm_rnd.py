"""
multi-parametric nonlinear program (mpNLP)
using Neuromancer's  differentiable parametric programming (DPP) approach

Objective: multi-dimensional rosenbrock
    see: https://en.wikipedia.org/wiki/Test_functions_for_optimization

subject to: multi-dimensional donut with random size

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import slim
import cvxpy as cp
from cvxpy import *

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable
from neuromancer.activations import activations
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import get_static_dataloaders
from neuromancer.loss import get_loss
from neuromancer.maps import Map
from neuromancer import blocks
from neuromancer.integers import IntegerProjection, IntegerInequalityProjection


def arg_mpMLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpMIMLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=50.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-nx_hidden", type=int, default=10,
           help="Number of hidden states of the solution map per problem dimension")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.003,
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
           help="number of decition variables and constraints")
    return parser


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpMLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    """
    # # #  Problem setup
    """
    n_x = 5*args.complexity           # number of decision variables
    n_theta = 1                  # number of parameters

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta
    nsim = 50000  # number of datapoints: increase sample density for more robust results
    samples = {"theta": np.random.uniform(low=2.0, high=6.0, size=(nsim, 1))}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpMINL problem formulation in Neuromancer
    """
    # components of a solution map
    func = blocks.MLP(insize=n_theta, outsize=n_x,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[args.nx_hidden] * args.n_layers)
    sol_map = Map(func,
            input_keys=["theta"],
            output_keys=["x"],
            name='primal_map')

    # variables
    x = Variable('x')  # decision variable as output from the solution map
    x_i = Variable('x')[:, :-1]
    x_j = Variable('x')[:, 1:]
    p = Variable('theta')  # sampled parametric variable

    # objective function
    f = (1-x_i)**2 + 2*(x_j-x_i**2)**2
    obj = f.minimize(weight=args.Q, name='obj')

    # non-convex constraints
    con_1 = args.Q_con * ((p/2)**2 <= x**2)
    con_2 = args.Q_con * (x**2 <= p**2)
    con_3 = args.Q_con * (x >= 0)

    # list of objectives, constraints, and components of solution map
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]

    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    problem = problem.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpNLP_rnd_2'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir,
                         verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpNLP_rnd_2'

    """
    # # #  mpNL problem solution in Neuromancer
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
        epochs=args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)


    """
    Performance evaluation
    """
    # sampled problem instance
    theta_i = np.random.uniform(low=2.0, high=6.0, size=(1))

    datapoint = {}
    datapoint['theta'] = torch.tensor([theta_i], dtype=torch.float)
    datapoint['name'] = "test"
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, :].detach().numpy()
    print(f'primal solution Neuromancer x={x_nm}')

    con_1_viol = np.maximum(0, (theta_i/2)**2 - x_nm**2)
    con_2_viol = np.maximum(0, x_nm**2 - theta_i**2)
    con_3_viol = np.maximum(0, -x_nm)
    con_viol = con_1_viol + con_2_viol + con_3_viol
    nm_con_viol_mean = np.mean(con_viol)
    x_i = x_nm[:-1]
    x_j = x_nm[1:]
    obj_value = np.sum((1 - x_i) ** 2 + 2 * (x_j - x_i ** 2) ** 2)
    print(f'Neuromancer mean constr. violation per variable {nm_con_viol_mean}')
    print(f'Neuromancer obj {obj_value}')