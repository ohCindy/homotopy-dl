"""
benchmark solution to QP problems with classical solvers
"""

import numpy as np
import torch
import cvxpy as cp
from cvxpy import *
import pandas as pd
import os
import sys

import neuromancer.arg as arg
from neuromancer.loggers import BasicLogger


def arg_mpMIMLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpMIMLP")
    gp.add("-complexity", type=int, default=5,
           help="number of decition variables and constraints")
    gp.add("-n_runs", type=int, default=3,
           help="number of evaluation runs")
    gp.add("-data_seed", type=int, default=30,
           help="Random seed used for simulated data")
    gp.add("-solver", type=str, default='OSQP',
           choices=['gurobi', 'OSQP'],
           help="select solver")
    return parser


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters

    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpMIMLP_problem()])
    args, grps = parser.parse_arg_groups()
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Problem setup
    """
    n_x = args.complexity                       # number of decision variables
    n_theta = int(np.ceil(args.complexity/2))          # number of parameters
    n_con = args.complexity                     # number of constraints

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    # fixed problem parameters
    A = torch.randn(n_x, n_con).t()
    b = torch.randn(n_con)
    E = args.complexity*torch.randn(n_theta, n_con).t()

    """
    CVXPY benchmark
    """
    # fixed problem parameters
    A = A.detach().detach().numpy()
    b = b.detach().numpy()
    E = E.detach().numpy()
    # variable in cvxpy
    x = cp.Variable(n_x)

# defining the problem constraints, objective and bounds
    def QP_param(theta_i, lb, ub):
        constr = [A@x <= b + E@theta_i,
                  x >= lb, x <= ub]
        obj = cp.Minimize(cp.sum(x ** 2))
        prob = cp.Problem(obj, constr)
        return prob

    """
    Performance comparison against solver in CVXPY and Gurobi
    """

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    samples = 100
    theta = np.random.uniform(low=3.0, high=5.0, size=(samples, n_theta))
    Int_viols = []
    Con_viols = []
    Objectives = []
    Sol_times = []
    time_limit = 3600  # time limit in seconds for the gurobi solver, 3600 = 1h
    for k in range(args.n_runs):
        theta_i = theta[k, :]
        print('\n -------------------------- ')
        print(f'Solve QP with scale {args.scale_factor} '
              f'using {args.solver}, eval run {k}')
        # Solve QP via CVXPY
        prob = MIQP_param(theta_i, -100, 100) # calling the problem with random parameter, lower and upper bounds on x

        if args.solver == 'OSQP':
            prob.solve(solver="OSQP", verbose=True)
        elif args.solver == 'gurobi':
            prob.solve(solver=cp.GUROBI, verbose=True,
                       MIPGap=0.01, TimeLimit=time_limit)

        # print(f'primal solution via solver x={x.value}')
        sol_con_viol = np.mean(np.maximum(0, A @ x.value - b - E@theta_i))
        objective_val = np.sum(x.value**2)
        Con_viols.append(sol_con_viol)
        Objectives.append(objective_val)
        Sol_times.append(prob._solve_time)
        print(f'solver mean constr. violation {sol_con_viol}')
        print(f'solver obj {objective_val}')
        print(f'solver time {prob._solve_time}')

    # save results to csv file
    save_file = False
    if save_file:
        dict = {'Con_viols': Con_viols, 'Objectives': Objectives, 'Sol_times': Sol_times}
        df = pd.DataFrame(dict)
        path = f'./QP_{args.solver}_runs'
        if not os.path.isdir(path):
            os.mkdir(path)
        df.to_csv(f'{path}/MIQP_scale_{args.scale_factor}.csv')

