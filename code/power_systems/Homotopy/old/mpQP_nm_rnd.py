"""
Tutorial script for formulating and solving multiparametric quadratic programs (mpQP)
using Neuromancer's data-driven differentiable parametric programming (DPP) approach

mpLP problem formulation:
    J(\Theta) = min_x ||x||^2
                s.t. Ax \le b + E*\Theta
                      \Theta \in F

    x = decision variable
    A, b, E = fixed problem parameters
    F = set of admissible parameters \Theta

The objective is to find an explicit optimizer (solution map) x = h(\Theta)
optimizing the value function J(\Theta)

We consider a mpLP problem with benchmark solution using Yalmip and MPT3 toolbox in Matlab
    https://yalmip.github.io/tutorial/multiparametricprogramming/

Further reading:
    Parametric programming
        https://en.wikipedia.org/wiki/Parametric_programming
        https://www.frontiersin.org/articles/10.3389/fceng.2020.620168/full
        http://www.optimization-online.org/DB_FILE/2018/04/6587.pdf
    Parametric linear programming
        https://scholar.rose-hulman.edu/cgi/viewcontent.cgi?article=1022&context=math_mstr
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/47907/10107_2005_Article_BF01581642.pdf?sequence=1
    Multiparametric (MPT3) toolbox - solving parametric programming problems in Matlab
        https://ieeexplore.ieee.org/abstract/document/6669862
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
from neuromancer.solvers import GradientProjection
from neuromancer.maps import Map
from neuromancer import blocks


def arg_mpQP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
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
    gp.add("-proj_grad", default=False, choices=[True, False],
           help="Whether to use projected gradient update or not.")
    gp.add("-complexity", type=int, default=5,
           help="number of decition variables and constraints")
    return parser


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpQP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    """
    # # #  QP problem matrices
    """

    # Parameters to change
    n_x = args.complexity                       # number of decision variables
    n_theta = int(np.ceil(args.complexity/2))          # number of parameters
    n_con = args.complexity                     # number of constraints

    # fixed problem parameters
    A = torch.randn(n_x, n_con).t()
    b = torch.randn(n_con)
    E = args.complexity*torch.randn(n_theta, n_con).t()

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    nsim = 50000  # number of datapoints: increase sample density for more robust results
    samples = {"theta": 0.5 * np.random.randn(nsim, n_theta)}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpQP problem formulation in Neuromancer
    """
    func = blocks.MLP(insize=n_theta, outsize=n_x,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[n_x*args.nx_hidden] * args.n_layers)
    sol_map = Map(func,
            input_keys=["theta"],
            output_keys=["x"],
            name='primal_map')

    """
    # # #  mpQP objective and constraints formulation in Neuromancer
    """
    # variables
    x = Variable('x')  # decision variable as output from the solution map
    theta = Variable('theta')  # sampled parametric variable
    # objective function
    loss = args.Q * (x == 0)^2  # weighted loss to be penalized
    loss.name = 'loss'
    # constraints: C = [ A*x <= b + E*theta, x >= 0]
    con_1 = args.Q_con * (x@A.t() <= b + theta@E.t())
    # con_2 = args.Q_con * (x >= 0)

    objectives = [loss]
    constraints = [con_1]
    components = [sol_map]

    if args.proj_grad:  # use projected gradient update
        project_keys = ["x"]
        projection = GradientProjection(constraints, project_keys)
        components.append(projection)

    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss, grad_inference=args.proj_grad)
    problem = problem.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpQP_rnd'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir,
                         verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpQP_rnd'

    """
    # # #  mpQP problem solution in Neuromancer
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
    # fixed problem parameters
    A = A.detach().detach().numpy()
    b = b.detach().numpy()
    E = E.detach().numpy()
    # single problem sample
    theta_i = 0.5 * np.random.randn(n_theta)

    # Solve QP via neuromancer for a single problem sample
    datapoint = {}
    datapoint['theta'] = torch.tensor([theta_i], dtype=torch.float)
    datapoint['name'] = "test"
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, :].detach().numpy()
    print(f'primal solution Neuromancer x={x_nm}')
    nm_con_viol = np.mean(np.maximum(0, A @ x_nm - b - E@theta_i))

    # KPIs to track besides training time
    print(f'Neuromancer mean constr. violation {nm_con_viol}')
    print(f'Neuromancer obj {np.sum(x_nm**2)}')


