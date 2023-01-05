"""
Power system Optimal Power Flow problem:
     min fobj s.t. g\leq 0, h=0
where fobj is usually the generation cost,
h contains the network balance equations (nonlinear eq constraints),
g contains some variables bounds (some are nonlinear)
problem can be formulated as:
    min_x=[Vr, Vi, Pg, Qg] \sum_i alpha_i*Pgi**2 + beta_i*Pgi
    s.t. network balance: Pgbus - Pd + j(Qgbus-Qd)= v conj(Yv),
            here Pd, Qd are vectors denoting load demands,
            Pg, Qg are active and reactive power generated from the generators
            Y is the bus admittance matrix (complex)
            v=Vr + j*Vi is a vector of bus voltages (complex)
            this equation will be split into real and imaginary parts when solving the problem
         Pg upper and lower bounds: Pgmin<=Pg<=Pgmax,
         Qg upper and lower bounds: Qgmin<=Qg<=Qgmax,
         voltage magnitude bounds: |V|min <= |V| <= |V|max
         reference bus angle should be zero: imag(v_ref) = 0
Here, we want to get
- add homotopy method to constrainted DL, for ACOPF problem
 - general homotopy heuristics
    1). Split and Shrink (SaS) for equality constraints h=0
        split: h = 0 split into two constraints 0 \leq h \leq 0
        this two ineq constraints can be relaxed to -e \leq h \leq e
        shrink: during homotpy process, we gradually reduce the relaxation by
            -(1-lambda_H)*e_H -e_L \leq h \leq (1-lambda_H)*e_H + e_L
    2). for inequality constraints g \leq 0
        i) Shrink bounds: g \leq 0 can be relaxed to g \leq epsilon,
           during the homotopy process, we can gradually decrease the purturbation by
                  g \leq (1-lambda_H) * epsilon
           so that it gets closer to the original constraints
        ii) Grow penalty:
            when using augmented loss function
            we can gradually increase the penalty strength of the violation
            by increasing the penalty term weights:
            e.g. Loss = fobj + lambda_H*w*||h||^2
    2) domain specific homotopy for power grid
        i) Tx stepping: gradually change the branches from short branches to the original branch
        2) load stepping: gradually increase the load demand from 0 to the original value,
        a trivial solution to the first homotopy problem can be easily available
        based on domain knowledge
        we can add a warmhomoloss to guide the initial update of the NN, towards the trivial solution


"""
import copy
import os
import os.path

import pypower
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader
import slim

# from neuromancer.trainer import Trainer
from trainer_new import Trainer
import trainer_new
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Loss
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import normalize_data, split_static_data, StaticDataset
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer import blocks
from neuromancer.maps import ManyToMany

# from neuromancer.plot import plot_loss_mpp, plot_solution_mpp

import cvxpy as cp
import acopfdisp as acopf

#
from pypower.api import ext2int
from pypower.ppoption import ppoption
from pypower.idx_brch import BR_R, BR_X
from pypower.api import makeYbus
from scipy.sparse import hstack, vstack, csr_matrix as sparse

# encoder-decoder setup:
from nm_extentions_for_opf import BatchNorm
import nm_extentions_for_opf
from nm_extentions_for_opf import param_scheduler
from nm_extentions_for_opf import Pypower_PF
from nm_extentions_for_opf import Wrapper


# correction
from neuromancer.solvers import GradientProjection


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
           help="choices: 0, 1; whether to train a new model or load existing models and print.")
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


def get_dataloaders(data, norm_type=None, split_ratio=None, num_workers=0, batch_size=None):
    """This will generate dataloaders for a given dictionary of data.
    Dataloaders are hard-coded for full-batch training to match NeuroMANCER's training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param norm_type: (str) type of normaleization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_static_data(data, split_ratio)

    train_data = StaticDataset(
        train_data,
        name="train",
    )
    dev_data = StaticDataset(
        dev_data,
        name="dev",
    )
    test_data = StaticDataset(
        test_data,
        name="test",
    )

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data) if not batch_size else batch_size,
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data) if not batch_size else batch_size,
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data) if not batch_size else batch_size,
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), train_data.dataset.dims


def get_loss(objectives, constraints, train_data, args):
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints, barrier=args.barrier_type)
    elif args.loss == 'augmented_lagrange':
        optimizer_args = {'inner_loop': args.inner_loop, "eta": args.eta, 'sigma': args.sigma,
                          'mu_init': args.mu_init, "mu_max": args.mu_max}
        loss = AugmentedLagrangeLoss(objectives, constraints, train_data, **optimizer_args)
    return loss


def main(args):
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    #
    casename = args.casename
    try:
        if casename == 'case9':
            from pypower.api import case9
            ppc_case = case9()
        elif casename == 'case14':
            from pypower.api import case14
            ppc_case = case14()
        elif casename == 'case30':
            from pypower.api import case30
            ppc_case = case30()
        elif casename == 'case39':
            from pypower.api import case39
            ppc_case = case39()
        elif casename == 'case57':
            from pypower.api import case57
            ppc_case = case57()
        elif casename == 'case118':
            from pypower.api import case118
            ppc_case = case118()
        elif casename == 'case300':
            from pypower.api import case300
            ppc_case = case300()
    except ValueError:
        print('Incorrect file name', sys.argv[1], 'Use option -casename <casename>',
              '\nSupported file names are case9,case39,case118,or case300')

    ppc = ext2int(ppc_case)
    out = acopf.setup(ppc)
    # print("pypower acopf setup:", out)

    Pd_alpha = out['Pd_alpha']
    nx = out['nx']
    nb = out['nb']
    ng = out['ng']
    alpha = out['alpha']
    betaT = out['betaT']
    baseMVA = out['baseMVA']

    # Set up Neuromancer solution
    nparam = 1

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    np.random.seed(args.data_seed)
    nsim = args.nsamples  # number of datapoints: increase sample density for more robust results

    samples = {"Pdtot": np.random.uniform(low=max(out['totPgmin'] * 1.1, out['totPd'] * args.lf_L),
                                          high=min(out['totPd'] * args.lf_H, out['totPgmax'] * 0.9),
                                          size=(nsim, nparam))}
    samples["Qdtot"] = samples["Pdtot"] / out['totPd'] * out['totQd']  # Qd changes according to Pd

    if args.loss == 'augmented_lagrange':
        batch_size = None  # current neuromancer Lagarangian method does not support minibatch
    elif args.loss == 'penalty':
        batch_size = 1024
    nstep_data, dims = get_dataloaders(samples, split_ratio=[70, 15], batch_size=batch_size)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  problem formulation in Neuromancer
    """
    # New syntax for maps
    activation = activations['relu']
    linmap = slim.maps['linear']

    bn = BatchNorm(insize=2)
    f1 = blocks.MLP(insize=2, outsize=nx,
                    bias=args.bias,
                    linear_map=linmap,
                    nonlin=activation,
                    hsizes=[args.nx_hidden] * args.n_layers)
    # initialization:
    torch.manual_seed(args.NN_seed)
    nm_extentions_for_opf.NN_initializer(f1, method='xavier')
    ##
    encoder = ManyToMany([bn],
                         input_keys=["Pdtot", "Qdtot"],
                         output_keys=["bnormed_in"],
                         name='encoder')
    # in_keys = encoder.output_keys
    sol_map = ManyToMany([f1],
                         input_keys=["bnormed_in"],  # ["Pdtot","Qdtot"] after batchnorm
                         output_keys=["x"],  # v=[Pg, Qg, Vr, Vi]
                         name='sol_map')
    # bounds = HardMinMaxBound(input_key_map={'x': 'x', 'xmax': 'xmax', 'xmin': 'xmin'},output_keys=['x'], name='bound')

    """write fobj and constraints for neuromancer"""
    # define vars
    x = Variable('x')  # acopf: x = [vr, vi, Pg, Qg], torch size (nsamples, nx)
    # print('x var size:', sol_map(train_data.dataset.get_full_batch())['x'].size())
    Pdtot = Variable('Pdtot')  # sampled parametric variable
    Qdtot = Variable('Qdtot')
    Vr = x[:, 0:nb]  # Vr, size (nsample, nb)V
    Vi = x[:, nb:2 * nb]  # Vi, size (nsample, nb)
    Pg = x[:, 2 * nb:2 * nb + ng]  # Pg, size (nsample, ng)
    Qg = x[:, 2 * nb + ng: nx]  # Qg, size (nsample, ng)
    """------------1. fobj--------------"""
    betacost = torch.from_numpy(out['betaT']).float().t().to(device)
    alphacost = torch.from_numpy(out['alpha']).float().to(device)
    opf_loss = Loss([f"x"], lambda x: torch.mean(torch.sum(((x @ alphacost) * x), dim=-1) + x @ betacost),
                    name='quad_loss')  # where is args.Q?
    opf_loss.weight = args.Q

    #

    """----------2. constraints ----------"""
    # 1) inequality constraint: variable bound
    # A_bnds = torch.from_numpy(out['A_bnds']).float()
    # b_bnds = torch.from_numpy(out['b_bnds']).float()
    # con_bnds = (x@A_bnds.t() <= b_bnds.t())
    Pgmax = torch.from_numpy(out['Pgmax']).float().to(device)
    Pgmin = torch.from_numpy(out['Pgmin']).float().to(device)
    Qgmax = torch.from_numpy(out['Qgmax']).float().to(device)
    Qgmin = torch.from_numpy(out['Qgmin']).float().to(device)
    Vm2max = torch.from_numpy(out['Vm2max']).float().to(device)
    Vm2min = torch.from_numpy(out['Vm2min']).float().to(device)
    con_Pgmax = (Pg <= Pgmax.t())
    con_Pgmin = (-Pg <= -Pgmin.t())
    con_Qgmax = (Qg <= Qgmax.t())
    con_Qgmin = (-Qg <= -Qgmin.t())
    con_Vm2max = (Vr ** 2 + Vi ** 2 <= Vm2max.t())
    con_Vm2min = (-Vr ** 2 - Vi ** 2 <= -Vm2min.t())
    if args.loss == 'penalty':
        con_Pgmax = args.Q_conineq * (con_Pgmax) ^ 2
        con_Pgmin = args.Q_conineq * (con_Pgmin) ^ 2
        con_Qgmax = args.Q_conineq * (con_Qgmax) ^ 2
        con_Qgmin = args.Q_conineq * (con_Qgmin) ^ 2
        con_Vm2max = args.Q_conineq * (con_Vm2max) ^ 2
        con_Vm2min = args.Q_conineq * (con_Vm2min) ^ 2
    else:
        con_Pgmax = args.Q_conineq * (con_Pgmax)
        con_Pgmin = args.Q_conineq * (con_Pgmin)
        con_Qgmax = args.Q_conineq * (con_Qgmax)
        con_Qgmin = args.Q_conineq * (con_Qgmin)
        con_Vm2max = args.Q_conineq * (con_Vm2max)
        con_Vm2min = args.Q_conineq * (con_Vm2min)
    con_Pgmax.name = 'ineq_Pg_upper_bnd'
    con_Pgmin.name = 'ineq_Pg_lower_bnd'
    con_Qgmax.name = 'ineq_Qg_upper_bnd'
    con_Qgmin.name = 'ineq_Qg_lower_bnd'
    con_Vm2max.name = 'ineq_Vm2_upper_bnd'
    con_Vm2min.name = 'ineq_Vm2_lower_bnd'
    # 2) equality constraints, network balance
    # Pg-Pd + i(Qg-Qd) = v(Yv)* => -Pd - iQd = v(Yv)* - Pg - iQg
    # split to real and imag part
    Ybus, Yf, Yt = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])  # topo: Ybus = Variable('topo')
    Ybus = sparse.toarray(Ybus)  # to dense numpy (nb,nb)
    Ybusr = torch.tensor(np.real(Ybus)).float().to(device)  # Ybus real part, torch tensor
    Ybusi = torch.tensor(np.imag(Ybus)).float().to(device)  # imag part
    Pd_a = torch.from_numpy(out['Pd_alpha']).float().to(device)  # Peach_load/P_total_load
    Qd_a = torch.from_numpy(out['Qd_alpha']).float().to(device)  # Peach_load/P_total_load
    neg_Cg = torch.from_numpy(sparse.toarray(out['neg_Cg'])).float().to(device)  # neg_Cg
    # con_complex = (Pg@out['neg_Cg'].t() + Qg@out['neg_Cg'].t()*1j + V_c*(V_c@Ybus.t()).conj() == -Pdtot*Pd_a.t() - Qdtot*Qd_a.t()*1j)
    con_Pbal = (Pg @ neg_Cg.t() +
                Vr * (Vr @ Ybusr.t()) -
                Vr * (Vi @ Ybusi.t()) +
                Vi * (Vi @ Ybusr.t()) +
                Vi * (Vr @ Ybusi.t()) == -Pdtot * Pd_a.t())
    con_Qbal = (Qg @ neg_Cg.t() +
                Vi * (Vr @ Ybusr.t()) -
                Vi * (Vi @ Ybusi.t()) -
                Vr * (Vi @ Ybusr.t()) -
                Vr * (Vr @ Ybusi.t()) == - Qdtot * Qd_a.t())
    if args.loss == 'penalty':
        con_Pbal = args.Q_coneq * (con_Pbal) ^ 2
        con_Qbal = args.Q_coneq * (con_Qbal) ^ 2
    else:
        con_Pbal = args.Q_coneq * (con_Pbal)
        con_Qbal = args.Q_coneq * (con_Qbal)
    con_Pbal.name = 'eq-Pbalance'
    con_Qbal.name = 'eq-Qbalance'
    # 3) equality constracon_ref = (x[:, nb+out["refs"]] == 0) #this worksint: reference bus
    con_ref = (x[:, nb + out["refs"]] == 0)  # this works
    # con_ref = (Vi[:,out["refs"]]==0) #this doesn't work
    if args.loss == 'penalty':
        con_ref = args.Q_coneq * (con_ref) ^ 2
    else:
        con_ref = args.Q_coneq * (con_ref)
    con_ref.name = 'eq-ref'
    # 2.5) Pull Pg up using inequality constraints: total Pg >= totalPd
    con_totPg = (Pdtot - Pg @ torch.ones((ng, 1)) <= -args.tol_Pgup)
    if args.loss == 'penalty':
        con_totPg = args.Q_Pgup * (con_totPg) ^ 2
    else:
        con_totPg = args.Q_Pgup * (con_totPg)
    con_totPg.name = 'ineq-totPg>totPd'

    """parameter tuning outerloops and homotopy"""
    Qstep = args.hyperstep if args.use_paramscheduler else args.epochs
    if args.use_paramscheduler:
        homo_param_sche = param_scheduler(Q0=0, stepsize=Qstep, mode='add', factor=args.homo_delta, Qmax=1,
                                          name='homotopy-parameter-(%s)' % args.coneq_homo_option)
    else:  # not necessarily needed just use it to clearly separate
        homo_param_sche = param_scheduler(Q0=1, stepsize=Qstep, factor=1, name='homotopy-parameter')

    # todo: find minoiuterloops, to be built in a hyper manager
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
        ppc_homo = copy.deepcopy(ppc)
        if args.use_paramscheduler:
            #
            if args.coneq_homo_option == 'Tx-stepping':
                # 2) rewrite equality constraints, network balance, by Tx-stepping
                R_br_ori = ppc_homo['branch'][:, BR_R]
                X_br_ori = ppc_homo['branch'][:, BR_X]
                R_br_homo = R_br_ori * lambda_H + (1 - lambda_H) * 0
                X_br_homo = X_br_ori * lambda_H + (1 - lambda_H) * 1e-4
                ppc_homo['branch'][:, BR_R] = R_br_homo
                ppc_homo['branch'][:, BR_X] = X_br_homo
            elif args.coneq_homo_option == 'Gmin-stepping':
                print("todo")  # todo
                raise 1
            elif args.coneq_homo_option == 'Load-stepping':
                Pd_a_ori = torch.from_numpy(out['Pd_alpha']).float()  # Peach_load/P_total_load
                Qd_a_ori = torch.from_numpy(out['Qd_alpha']).float()  # Peach_load/P_total_load
                Pd_a = Pd_a_ori * lambda_H
                Qd_a = Qd_a_ori * lambda_H
                Pd_a = Pd_a.to(device)
                Qd_a = Qd_a.to(device)
                #
                con_totPg = (Pdtot * lambda_H - Pg @ torch.ones((ng, 1)) <= -args.tol_Pgup)
                if args.loss == 'penalty':
                    con_totPg = 10 * args.Q_conineq * (con_totPg) ^ 2
                else:
                    con_totPg = 10 * args.Q_conineq * (con_totPg)
                con_totPg.name = 'ineq-totPg>totPd'
                #
                warmhomo_loss = Loss([f"x"], lambda x: torch.mean(torch.sum((x - 1e-6) ** 2, dim=-1)),
                                     name='warmhomo_loss')
                warmhomo_loss.weight = args.Q_warmhomo if lambda_H == 0 else 0.0
                print(args.coneq_homo_option, "warmhomo_loss.weight ", warmhomo_loss.weight)
        Ybus, Yf, Yt = makeYbus(ppc_homo['baseMVA'],
                                ppc_homo['bus'],
                                ppc_homo['branch'])  # topo: Ybus = Variable('topo')
        Ybus = sparse.toarray(Ybus)  # to dense numpy (nb,nb)
        Ybusr = torch.tensor(np.real(Ybus)).float().to(device)  # Ybus real part, torch tensor
        Ybusi = torch.tensor(np.imag(Ybus)).float().to(device)  # imag part
        # con_complex = (Pg@out['neg_Cg'].t() + Qg@out['neg_Cg'].t()*1j + V_c*(V_c@Ybus.t()).conj() == -Pdtot*Pd_a.t() - Qdtot*Qd_a.t()*1j)
        con_Pbal = (Pg @ neg_Cg.t() +
                    Vr * (Vr @ Ybusr.t()) -
                    Vr * (Vi @ Ybusi.t()) +
                    Vi * (Vi @ Ybusr.t()) +
                    Vi * (Vr @ Ybusi.t()) == -Pdtot * Pd_a.t())
        con_Qbal = (Qg @ neg_Cg.t() +
                    Vi * (Vr @ Ybusr.t()) -
                    Vi * (Vi @ Ybusi.t()) -
                    Vr * (Vi @ Ybusr.t()) -
                    Vr * (Vr @ Ybusi.t()) == - Qdtot * Qd_a.t())
        if args.loss == 'penalty':
            con_Pbal = args.Q_coneq * (con_Pbal) ^ 2
            con_Qbal = args.Q_coneq * (con_Qbal) ^ 2
        else:
            con_Pbal = args.Q_coneq * (con_Pbal)
            con_Qbal = args.Q_coneq * (con_Qbal)
        con_Pbal.name = 'eq-Pbalance'
        con_Qbal.name = 'eq-Qbalance'
        # general homotopy SaS: split eq con to 2 ineq con, and shrink
        con_Pbal_right = (Pg @ neg_Cg.t() +
                          Vr * (Vr @ Ybusr.t()) -
                          Vr * (Vi @ Ybusi.t()) +
                          Vi * (Vi @ Ybusr.t()) +
                          Vi * (Vr @ Ybusi.t()) <= -Pdtot * Pd_a.t() +
                          (1 - lambda_H) * args.con_res_H + args.con_res_L)
        con_Pbal_left = (Pg @ neg_Cg.t() +
                         Vr * (Vr @ Ybusr.t()) -
                         Vr * (Vi @ Ybusi.t()) +
                         Vi * (Vi @ Ybusr.t()) +
                         Vi * (Vr @ Ybusi.t()) >= -Pdtot * Pd_a.t() -
                         (1 - lambda_H) * args.con_res_H - args.con_res_L)
        con_Qbal_right = (Qg @ neg_Cg.t() +
                          Vi * (Vr @ Ybusr.t()) -
                          Vi * (Vi @ Ybusi.t()) -
                          Vr * (Vi @ Ybusr.t()) -
                          Vr * (Vr @ Ybusi.t()) <= - Qdtot * Qd_a.t() +
                          (1 - lambda_H) * args.con_res_H + args.con_res_L)
        con_Qbal_left = (Qg @ neg_Cg.t() +
                         Vi * (Vr @ Ybusr.t()) -
                         Vi * (Vi @ Ybusi.t()) -
                         Vr * (Vi @ Ybusr.t()) -
                         Vr * (Vr @ Ybusi.t()) >= - Qdtot * Qd_a.t() -
                         (1 - lambda_H) * args.con_res_H - args.con_res_L)
        if args.loss == 'penalty':
            con_Pbal_left = args.Q_coneq * (con_Pbal_left) ^ 2
            con_Pbal_right = args.Q_coneq * (con_Pbal_right) ^ 2
            con_Qbal_left = args.Q_coneq * (con_Qbal_left) ^ 2
            con_Qbal_right = args.Q_coneq * (con_Qbal_right) ^ 2
        else:
            con_Pbal_left = args.Q_coneq * (con_Pbal_left)
            con_Pbal_right = args.Q_coneq * (con_Pbal_right)
            con_Qbal_left = args.Q_coneq * (con_Qbal_left)
            con_Qbal_right = args.Q_coneq * (con_Qbal_right)
        con_Pbal_left.name = 'eq-Pbalance-left'
        con_Qbal_left.name = 'eq-Qbalance-left'
        con_Pbal_right.name = 'eq-Pbalance-right'
        con_Qbal_right.name = 'eq-Qbalance-right'
        # rewrite bnds constraints with homo bnds when needed
        if args.add_bnds_homo:
            Pgmax_homo = Pgmax + (1 - lambda_H) * args.Pgmax_homofactor
            Pgmin_homo = Pgmin * lambda_H
            Qgmax_homo = Qgmax + (1 - lambda_H)
            Qgmin_homo = Qgmin - (1 - lambda_H)
            Vm2max_homo = Vm2max
            Vm2min_homo = Vm2min * lambda_H
            con_Pgmax = (Pg <= Pgmax_homo.t())
            con_Pgmin = (-Pg <= -Pgmin_homo.t())
            con_Qgmax = (Qg <= Qgmax_homo.t())
            con_Qgmin = (-Qg <= -Qgmin_homo.t())
            con_Vm2max = (Vr ** 2 + Vi ** 2 <= Vm2max_homo.t())
            con_Vm2min = (-Vr ** 2 - Vi ** 2 <= -Vm2min_homo.t())
            if args.loss == 'penalty':
                con_Pgmax = args.Q_conineq * (con_Pgmax) ^ 2
                con_Pgmin = args.Q_conineq * (con_Pgmin) ^ 2
                con_Qgmax = args.Q_conineq * (con_Qgmax) ^ 2
                con_Qgmin = args.Q_conineq * (con_Qgmin) ^ 2
                con_Vm2max = args.Q_conineq * (con_Vm2max) ^ 2
                con_Vm2min = args.Q_conineq * (con_Vm2min) ^ 2
            else:
                con_Pgmax = args.Q_conineq * (con_Pgmax)
                con_Pgmin = args.Q_conineq * (con_Pgmin)
                con_Qgmax = args.Q_conineq * (con_Qgmax)
                con_Qgmin = args.Q_conineq * (con_Qgmin)
                con_Vm2max = args.Q_conineq * (con_Vm2max)
                con_Vm2min = args.Q_conineq * (con_Vm2min)
            con_Pgmax.name = 'ineq_Pg_upper_bnd'
            con_Pgmin.name = 'ineq_Pg_lower_bnd'
            con_Qgmax.name = 'ineq_Qg_upper_bnd'
            con_Qgmin.name = 'ineq_Qg_lower_bnd'
            con_Vm2max.name = 'ineq_Vm2_upper_bnd'
            con_Vm2min.name = 'ineq_Vm2_lower_bnd'
        """adjust constraint weight based on all homotopy"""
        #
        con_totPg.weight = args.Q_Pgup  # must satisfy this in every loop
        con_ref.weight = args.Q_coneq  #
        #
        # con_Pbal.weight = args.Q_coneq
        # con_Qbal.weight = args.Q_coneq
        con_Pgmax.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)
        con_Pgmin.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)
        con_Qgmax.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)
        con_Qgmin.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)
        con_Vm2max.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)
        con_Vm2min.weight = args.Q_conineq * max(lambda_H, args.add_bnds_homo)

        ##
        objs = [opf_loss]
        cons = [con_Pbal, con_Qbal,
                con_ref,
                con_Pgmax, con_Pgmin,
                con_Qgmax, con_Qgmin,
                con_Vm2max, con_Vm2min, ]

        if args.use_paramscheduler:
            if args.coneq_homo_option == 'Load-stepping':
                objs.append(warmhomo_loss)
            elif args.coneq_homo_option == 'SaS':
                cons = [con_Pbal_left, con_Pbal_right, con_Qbal_left, con_Qbal_right,
                        con_ref,
                        con_Pgmax, con_Pgmin,
                        con_Qgmax, con_Qgmin,
                        con_Vm2max, con_Vm2min, ]

        if args.add_pullup:
            cons.append(con_totPg)

        objectives, constraints = objs, cons  # write_objs_and_cons(out, args)
        components = [encoder, sol_map]

        #
        if args.add_proj:  # use projected gradient update
            project_keys = ["x"]
            if args.proj_option == "all":
                cons_proj = cons
            elif args.proj_option == "ineq":
                cons_proj = [con_Pgmax, con_Pgmin,
                             con_Qgmax, con_Qgmin,
                             con_Vm2max, con_Vm2min, ]
            elif args.proj_option == "eq":
                cons_proj = [con_Pbal, con_Qbal, con_ref]
            else:
                print("nothing for projection, check proj_optin again")
                raise
            projection = GradientProjection(cons_proj, input_keys=project_keys,
                                            num_steps=5, name='proj_correction')
            components.append(projection)
            print("Projection layer added to the components")

        # Create constrained optimization loss
        # Create constrained optimization loss
        loss = get_loss(objectives, constraints, train_data, args)
        # constrained optimization problem construction
        model = Problem(components, loss, grad_inference=bool(args.add_proj))
        model = model.to(device)

        """
        # # # Metrics and Logger
        """

        savedir = args.savedir + '/Outerloop%d' % current_epoch if args.use_paramscheduler else args.savedir
        args.verbosity = 1
        metrics = ["dev_loss"]
        logger = BasicLogger(args=args, savedir=savedir, verbosity=args.verbosity, stdout=metrics)
        logger.args.system = 'mpACOPF'

        """
        # # #  problem solution in Neuromancer
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # define trainer
        trainer = Trainer(
            model,
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
        #
        trainer.current_epoch = current_epoch * Qstep
        # Train model
        is_train = args.is_train
        if not is_train:
            try:
                model_dir = savedir + '/best_model_state_dict.pth'
                print("attempting to load model from ", model_dir)
                best_model = torch.load(model_dir)
                model.load_state_dict(best_model)
            except FileNotFoundError:
                print("model in this outerloop not found, train one, still attempt to load model in the next outerloop")
                is_train = 1
        if is_train:
            best_model = trainer.train()

        best_outputs = trainer.test(best_model)

        """test the method on test data, and calculate metrics"""
        #
        ntest = 100  # number of test samples, if ntest=1, we just test on the base load of ppc_case
        # params is created on cpu
        if ntest == 1:
            # the base case load
            params = {"Pdtot": torch.tensor([out['totPd']]).view(1, 1).float(),
                      "Qdtot": torch.tensor([out['totQd']]).view(1, 1).float(),
                      "Pds": torch.tensor([out['totPd'] * out['Pd_alpha']]).view(1, nb).float(),
                      "Qds": torch.tensor([out['totQd'] * out['Qd_alpha']]).view(1, nb).float(), }
        else:
            # test dataset
            params = test_data.dataset.get_full_batch()
            params['Pdtot'] = params['Pdtot'][0:ntest]
            params['Qdtot'] = params['Qdtot'][0:ntest]
            params["Pds"] = (params['Pdtot'] * out['Pd_alpha'].T).float()
            params["Qds"] = (params['Qdtot'] * out['Qd_alpha'].T).float()
        ntest = len(params["Pdtot"])
        # send data to device and calculate Neuromancer solutions
        x_neuromancer = model.step(trainer_new.move_batch_to_device(params, device=device))  # on device
        # x_optim = x_neuromancer['x'].detach().numpy().reshape((ntest, nx))
        x_optim = x_neuromancer['x'].detach().view(ntest, nx)
        V_optim = x_optim[:, 0:2 * nb]  # (ntest, 2nb)rectangular
        Pg_optim = x_optim[:, 2 * nb:2 * nb + ng]  # (ntest, ng) Pg
        Vmag_optim = torch.abs(V_optim[:, 0:nb] + 1j * V_optim[:, nb:2 * nb])
        Vgmag_optim = -Vmag_optim @ neg_Cg # (ntest, ng) |Vg|_nm
        #
        baseMVA = out['baseMVA']
        # Objective value
        obj_n = torch.sum(((x_optim @ alphacost) * x_optim), dim=-1) + x_optim @ betacost
        ## pypower solver to get golden standard
        test_label_file = os.path.dirname(args.savedir) + '/data/' + str(args.casename) + '_xoptimpyp_ntest%d.npy' % ntest
        print(test_label_file)
        if current_epoch == 0:
            if device != "cpu":
                x_optim_pyp = np.zeros((ntest, nx))
                print("skip pypower running on gpu")
            else:
                try:
                    x_optim_pyp = np.load(test_label_file)
                except:
                    x_optim_pyp = nm_extentions_for_opf.getlabels_acopf_pyp(ppc_case,
                                                                            params['Pds'], params['Qds'])
            Pg_optim_pyp = x_optim_pyp[:, 2 * nb:2 * nb + ng]  # (ntest, ng)
            Qg_optim_pyp = x_optim_pyp[:, 2 * nb + ng:nx]  # (ntest, ng)
            V_optim_pyp = x_optim_pyp[:, 0:nb] + 1j * x_optim_pyp[:, nb:2 * nb]  # (ntest, nb) complex
            Vgm_optim_pyp = -np.abs(V_optim_pyp) @ neg_Cg.cpu().data.numpy()  # (ntest, ng)
            # calculate the obj of pypower solution, numpy, cpu
            obj_pyp = (x_optim_pyp ** 2) @ (np.diag(alpha).reshape(-1, 1)) + x_optim_pyp @ betaT.T  # (ntest, 1)

        # #
        # idx_test = 0  # pick one sample to print solution
        # print("Neuromancer solution: Pg", Pg_optim[idx_test] * baseMVA, '|Vg| ', Vgmag_optim[idx_test])
        # print("pypower solution: Pg ", Pg_optim_pyp[idx_test] * baseMVA, '|Vg| ', Vgm_optim_pyp[idx_test])

        # check violation of eq constraints
        x_test = x_neuromancer['x'].detach().view(ntest, nx)  # on device
        Pd_test = params["Pds"]  # on cpu
        Qd_test = params["Qds"]  # on cpu

        # check PQ balance violations of the neuromancer output on the orignal problem case
        res_Pbal, res_Qbal = acopf.check_PQbal(x_test,
                                               ppc, Pd_test, Qd_test,
                                               neg_Cg, nb, ng, nx)
        ave_Pbal = torch.abs(res_Pbal).mean(dim=-1)  # (ntest,)
        ave_Pbal_mean = ave_Pbal.mean().detach().numpy().item()  # mean over test data
        ave_Pbal_std = ave_Pbal.std().detach().numpy().item()  # std over test data
        ave_Qbal = torch.abs(res_Qbal).mean(dim=-1)
        ave_Qbal_mean = ave_Qbal.mean().detach().numpy().item()
        ave_Qbal_std = ave_Qbal.std().detach().numpy().item()

        max_Pbal, _ = torch.abs(res_Pbal).max(dim=-1)
        max_Pbal_mean = max_Pbal.mean().detach().numpy().item()
        max_Pbal_std = max_Pbal.std().detach().numpy().item()
        max_Qbal, _ = torch.abs(res_Qbal).max(dim=-1)
        max_Qbal_mean = max_Qbal.mean().detach().numpy().item()
        max_Qbal_std = max_Qbal.std().detach().numpy().item()
        #
        ave_bal = (ave_Pbal + ave_Qbal) / 2
        ave_bal_mean = ave_bal.mean().detach().numpy().item()
        ave_bal_std = ave_bal.std().detach().numpy().item()
        max_bal = torch.maximum(max_Pbal, max_Qbal)
        max_bal_mean = max_bal.mean().detach().numpy().item()
        max_bal_std = max_bal.std().detach().numpy().item()
        n_Pbal_viol = (torch.abs(res_Pbal) > 1e-2).sum(dim=-1)  # (ntest,)
        n_Qbal_val = (torch.abs(res_Qbal) > 1e-2).sum(dim=-1)  # (ntest,)
        # check violation of bnds constraints on neuromancer output
        res_bnds, \
        n_Pgmax_viol, n_Pgmin_viol, n_Qgmax_viol, n_Qgmin_viol, \
        n_Vm2max_viol, n_Vm2min_viol = acopf.check_bnds(x_test,
                                                        Pgmax, Pgmin, Qgmax, Qgmin,
                                                        Vm2max, Vm2min,
                                                        nb, ng, nx)
        ave_bnds_viol = torch.abs(res_bnds).mean(dim=-1)
        ave_bnds_viol_mean = ave_bnds_viol.mean().detach().numpy().item()
        ave_bnds_viol_std = ave_bnds_viol.std().detach().numpy().item()
        max_bnds_viol, _ = torch.abs(res_bnds).max(dim=-1)
        max_bnds_viol_mean = max_bnds_viol.mean().detach().numpy().item()
        max_bnds_viol_std = max_bnds_viol.std().detach().numpy().item()
        n_ineq_viol = n_Pgmax_viol + n_Pgmin_viol + \
                      n_Qgmax_viol + n_Qgmin_viol + \
                      n_Vm2max_viol + n_Vm2min_viol
        n_ineq_viol_mean = n_ineq_viol.float().mean()
        n_ineq_viol_std = n_ineq_viol.float().std()
        ntest_close = (n_Pbal_viol + n_Qbal_val + n_ineq_viol == 0).sum()
        print("Check raw NN output:")
        print("    %d/%d samples are close enough " % (ntest_close, ntest))
        print("    (mean) nm Objective = ", obj_n.mean())
        print("    (mean) pyp Objective= ", obj_pyp.mean())
        print(
            "    eq_viol mean %.4f (%.4f), max %.4f (%.4f); mean P %.4f (%.4f), Q %.4f (%.4f), (max): P %.4f (%.4f), Q %.4f (%.4f)"
            % (ave_bal_mean, ave_bal_std, max_bal_mean, max_bal_std,
               ave_Pbal_mean, ave_Pbal_std, ave_Qbal_mean, ave_Qbal_std,
               max_Pbal_mean, max_Pbal_std, max_Qbal_mean, max_Qbal_std))
        print("    averagely %d P_bal viols, %d Q_bal viols, (node res>0.01)"
              % (n_Pbal_viol.float().mean(), n_Qbal_val.float().mean()))
        print("    bnds_viol mean %.4f (%.4f), max %.4f (%.4f); "
              % (ave_bnds_viol_mean, ave_bnds_viol_std, max_bnds_viol_mean, max_bnds_viol_std))
        print(
            "    (nm) averagely %d(%.4f) ineq violations, n_ineq_violsx: %d Pgmax, %d Pgmin, %d Qgmax, %d Qgmin %d |V|max, %d |V|min"
            % (n_ineq_viol_mean, n_ineq_viol_std,
               n_Pgmax_viol.float().mean(), n_Pgmin_viol.float().mean(),
               n_Qgmax_viol.float().mean(), n_Qgmin_viol.float().mean(),
               n_Vm2max_viol.float().mean(), n_Vm2min_viol.float().mean()))
        #
        if max_bal_mean < best_res:
            best_res = max_bal_mean
            #
            best_obj = obj_n.mean()
            best_max_bal_mean, best_max_bal_std = max_bal_mean, max_bal_std
            best_ave_bal_mean, best_ave_bal_std = ave_bal_mean, ave_bal_std
            best_ave_bnds_viol_mean, best_ave_bnds_viol_std, best_max_bnds_viol_mean, best_max_bnds_viol_std = \
                ave_bnds_viol_mean, ave_bnds_viol_std, max_bnds_viol_mean, max_bnds_viol_std
            # best_max_Pbal_mean, best_max_Pbal_std, best_max_Qbal_mean, best_max_Qbal_std = \
            #     max_Pbal_mean, max_Pbal_std, max_Qbal_mean, max_Qbal_std
            best_n_ineq_viol_mean, best_n_ineq_viol_std = n_ineq_viol_mean, n_ineq_viol_std
            best_nclose = ntest_close

        #
        opf_decision = [(Pg_optim * baseMVA).tolist(),
                        Vgmag_optim.tolist()]  # np.concatenate((Pg_optim*baseMVA, Vgmag_optim),axis=1).tolist()
        nm_sol_rec.append(opf_decision)  # nm_sol_rec (nstep, 2, ntest, ng)
        this_metric = [obj_n.mean(), obj_pyp.mean(),
                       ntest_close,
                       ave_bal_mean, ave_bal_std, max_bal_mean, max_bal_std,
                       ave_bnds_viol_mean, ave_bnds_viol_std, max_bnds_viol_mean, max_bnds_viol_std,
                       n_ineq_viol_mean, n_ineq_viol_std, ]
        metric_rec.append(this_metric)


    print("==============Summary Printings===================")
    print(args.savedir)

    metric_names = ['mean obj_nm', 'obj_pyp',
                    'nclose',
                    'eq (mean viol)', 'std', '(max viol)', 'std',
                    'bnds (mean viol)', 'std', '(max viol)', 'std',
                    '(n_viol) bnds', 'std', ]
    metric_formats = ['%.2f', '%.2f',
                      '%d',
                      '%.4f', '%.4f', '%.4f', '%.4f',
                      '%.4f', '%.4f', '%.4f', '%.4f',
                      '%.3f', '%.3f', ]
    print("check violations:")
    if ntest == 1:
        for istep in range(len(nm_sol_rec)):
            print("Outer:", istep,
                  'Pg (tot %.3f)' % np.sum(nm_sol_rec[istep][0][-1]),
                  ['{0:.2f}'.format(val) for val in nm_sol_rec[istep][0][-1]],
                  ['|Vg| {0:.3f}'.format(val) for val in nm_sol_rec[istep][1][-1]],
                  )
        print("Pypower  Pg (tot %.2f)" % Pg_optim_pyp[-1].sum(),
              ['{0:.2f}'.format(val) for val in Pg_optim_pyp[-1]],
              ['|Vg| {0:.3f}'.format(val) for val in Vgm_optim_pyp[-1]])
        print("total Pd", params['Pdtot'] * baseMVA)
    for istep in range(len(nm_sol_rec)):
        print("Outer:", istep,
              [('%s ' + metric_formats[imetric]) % (metric_names[imetric], val) for imetric, val in
               enumerate(metric_rec[istep])],
              )

    out_dict = dict()
    out_dict['metric_names'] = metric_names
    out_dict['metric_formats'] = metric_formats
    out_dict['nm_sol_rec'] = nm_sol_rec  # (nouterloop, ntypeofsol, nsample, ng
    out_dict['metric_rec'] = metric_rec  # (nouterloop, nmetric)
    out_dict['best_nclose'] = best_nclose
    out_dict['n_outerloop'] = current_epoch + 1
    out_dict['nb'] = nb
    out_dict['ng'] = ng
    out_dict['nx'] = nx

    # plot
    performance_tag = ('_%dclose' % out_dict['best_nclose'] if out_dict['best_nclose'] > 0 else '')
    title = ('obj %.2f, eq me %.4f(%.4f),max %.3f(%.3f),bnds me %.4f(%.4f),max %.4f(%.4f)') \
            % (best_obj,
               best_ave_bal_mean, best_ave_bal_std, best_max_bal_mean, best_max_bal_std,
               best_ave_bnds_viol_mean, best_ave_bnds_viol_std, best_max_bnds_viol_mean, best_max_bnds_viol_std,
               )
    out_dict['performance'] =  title + '\n'



    print('\n', args.savedir)
    print(title)
    return out_dict


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpACOPF_problem()])
    args, grps = parser.parse_arg_groups()
    #
    tag = nm_extentions_for_opf.get_tag_acopf(args)
    print("validate tag ", tag)
    if args.savedir == "test": #"test" is the built-in default savedir
        args.savedir = "test_mpACOPF_homotopy" + '/%s/%s' % (args.casename, tag)
    print("model save directory", args.savedir)
    #
    out_dict = main(args)
    from datetime import date
    today = date.today()
    with open(os.path.dirname(args.savedir) + '/result_opf.txt', 'a') as f: #with open(args.savedir + '/result.txt', 'w') as f:
        print('\n')
        f.write(str(today) + ' ' + args.savedir + '\n')
        f.write(tag + '\n')
        f.write(out_dict['performance'])
    print(os.path.dirname(args.savedir) + '/result_opf.txt')


    markend = 1



"""
================define functions for meta learning
"""