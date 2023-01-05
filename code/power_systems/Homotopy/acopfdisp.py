import numpy as np
import pypower
from pypower.api import makeYbus
from pypower.idx_bus import PD,QD,BUS_TYPE,REF,GS,VM, VA, VMAX, VMIN
from pypower.idx_gen import PG, PMIN,PMAX, QMAX, QMIN, GEN_STATUS,GEN_BUS
from pypower.idx_brch import F_BUS, T_BUS, RATE_A, BR_STATUS
from scipy.sparse import hstack, vstack,csr_matrix as sparse
from numpy import flatnonzero as find
from numpy import ones, arange, Inf

import cvxpy as cp




def setup(ppc):

    # baseMVA,bus, gen, and branch, gencost
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    gencost = ppc['gencost']

    out = {}
    out['baseMVA'] = baseMVA
    
    # sizes
    nb = bus.shape[0]    ## number of buses
    nl = branch.shape[0] ## number of branches
    ng = gen.shape[0]    ## number of generators
    nx = 2*nb + 2*ng         ## Total variables
    #final x = [vm, va, Pg, Qg], no longer used, because the toolbox doesn't support sin(),cos()
    #final x= [vr, vi, Pg, Qg]

    #make admittance matrix
    # Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    # out['Ybus_base'] = Ybus
    # out['Yf_base'] = Yf
    # out['Yt_base'] = Yt

    # reference bus
    refs = find(bus[:,BUS_TYPE] == REF)
    out["refs"] = refs

    # Load
    Pd = ppc['bus'][:,PD]/baseMVA
    Qd = ppc['bus'][:,QD]/baseMVA
    # Total load
    Pdsum = np.sum(Pd)
    totPd = sum(Pd) #total load power
    totQd = sum(Qd)  # total load power
    # size of load
    nd = len(Pd)

    out['totPd'] = totPd
    out['totQd'] = totQd
    # A note on loads. We will set the loads as Pdsum*Pdi_alpha where Pdsum is the total load
    # and Pdi_alpha is the proportion for load Pdi.
    Pd_alpha = (Pd/totPd).reshape((nb,1))
    Qd_alpha = (Qd/totQd).reshape((nb,1))

    out['Pd_alpha'] = Pd_alpha
    out['Qd_alpha'] = Qd_alpha

    # Initial guess
    # Voltage magnitude and angle
    Vm = bus[:, VM].reshape((nb, 1))
    Va = bus[:,VA].reshape((nb,1))*np.pi/180.0
    Vr = Vm*np.cos(Va)
    Vi = Vm*np.sin(Va)

    # Generator power initial guess
    Pg = 0.5*(gen[:,PMIN] + gen[:,PMAX]).reshape((ng,1))/baseMVA
    Qg = 0.5*(gen[:,QMIN] + gen[:,QMAX]).reshape((ng,1))/baseMVA

    x0 = np.zeros((nx,1))
    x0[0:nb] = Vr #Vm
    x0[nb:2*nb] = Vi #Va
    x0[2*nb:2*nb+ng] = Pg
    x0[2*nb+ng : nx] = Qg

    out['x0'] = x0

    # Bounds
    # Voltage magnitude
    Vmmax = bus[:, VMAX].reshape((nb,1)) #pu
    Vmmin = bus[:, VMIN].reshape((nb,1))
    Vm2max = Vmmax**2 # upper bound of |V|^2
    Vm2min = Vmmin**2 #|V|^2 min
    out["Vm2max"] = Vm2max
    out["Vm2min"] = Vm2min
    #phase angle, no longer used
    Vamax = (100.0*ones(nb)).reshape((nb,1))
    Vamin = -Vamax
    Vamin[refs] = Va[refs]
    Vamax[refs] = Va[refs]
    
    # Total online min. and max. generation
    Pgmin = gen[:,PMIN].reshape((ng,1))/baseMVA
    Pgmax = gen[:,PMAX].reshape((ng,1))/baseMVA
    Qgmin = gen[:, QMIN].reshape((ng, 1)) / baseMVA
    Qgmax = gen[:, QMAX].reshape((ng, 1)) / baseMVA
    out["Pgmin"] = Pgmin
    out["Pgmax"] = Pgmax
    out["Qgmin"] = Qgmin
    out["Qgmax"] = Qgmax


    # for equality constraints: Pg-Pd + i(Qg-Qd) = v(Yv)* => -Pd - iQd = v(Yv)* - Pg - iQg
    neg_Cg = sparse((-np.ones(ng), (ppc['gen'][:, GEN_BUS], np.arange(ng))), (nb, ng))
    #(nb,ng) matrix: neg_Cg dot Pg = -Pg(bus length)
    out['neg_Cg'] = neg_Cg

    # Inequality constraints
    ## bound constraints xmin<=x<=xmax, to be used as soft constraints
    ## designed for x= [Vm ,Va, Pg, Qg]
    # xmin = np.concatenate((Vmmin,Vamin,Pgmin,Qgmin))
    # xmax = np.concatenate((Vmmax,Vamax,Pgmax,Qgmax))
    # out['xmin'] = xmin
    # out['xmax'] = xmax
    # bbounds = np.zeros((2*nx,1))
    # Abounds = np.zeros((2*nx,nx))
    # Abounds[0:nx,:] = np.diag(-np.ones(nx))
    # bbounds[0:nx] = -xmin
    # Abounds[nx:2*nx,:] = np.diag(np.ones(nx))
    # bbounds[nx:2*nx] = xmax
    # out['A_bnds'] = Abounds
    # out['b_bnds'] = bbounds
    # usage: Abounds@x <= bbounds

    ## branch flow constraints
    il = find((branch[:, BR_STATUS] != 0) & (branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    nl2 = len(il)  ## number of constrained lines
    """todo: store some branch idx info? from buses, to buses?"""
    nl2 = len(il)         ## number of constrained lines
    upf = (branch[il, RATE_A] / baseMVA ).reshape((nl2,1))
    upt = (branch[il, RATE_A] / baseMVA ).reshape((nl2,1))
    # Relax flow bounds
    #upf = 10000.0*np.ones((nl2,1))
    #upt = 10000.0*np.ones((nl2,1))
    lpf = -upf
    lpt = -upt
    # out["Fmax_from"] = upf #upper bound of branch power flow, at from side
    # out["Fmin_from"] = lpf
    # out["Fmax_to"] = upt
    # out["Fmin_to"] = lpt
    out["S2max_from"] = upf**2 #upper bound of S^2, (the square of apparent power) at from side
    #out["S2min_from"] = -lpf**2 #lower bound not needed since S^2 must >=0
    out["S2max_to"] = upt**2
    #out["S2min_to"] = -lpt**2
    Cf = sparse((np.ones(nl2), (np.arange(nl2), ppc['branch'][:, F_BUS])), (nl2, nb))
    Ct = sparse((np.ones(nl2), (np.arange(nl2), ppc['branch'][:, T_BUS])), (nl2, nb))
    out["Cf"] = Cf #(nl2, nb) matrix, Vf = Cf dot v
    out["Ct"] = Ct #(nl2, nb) matrix, Vt = Ct dot v
    
    # Cost function, x = [vm, va, pg, qg]
    alpha = np.zeros((nx,nx))
    alpha[2*nb:2*nb+ng,2*nb:2*nb+ng] = np.diag(gencost[:,4])*baseMVA*baseMVA
    #alpha = np.diag(gencost[:,4]) #if only considering Pg variables

    betaT = np.concatenate((np.zeros((1,2*nb)),
                            gencost[:,5].reshape((1,ng)),
                            np.zeros((1,ng))),axis=1)*baseMVA
    #gamma = gencost[:,6]

    out['alpha'] = alpha
    out['betaT']  = betaT
    
    totPgmin = np.sum(Pgmin)
    totPgmax = np.sum(Pgmax)
    totQgmin = np.sum(Qgmin)
    totQgmax = np.sum(Qgmax)
    out['totPgmin'] = totPgmin
    out['totPgmax'] = totPgmax
    out['totQgmin'] = totQgmin
    out['totQgmax'] = totQgmax

    neqcon   = 2*nb  # number of equality constraints
    nineqcon = 4*nl2 + 2*nb + 4*ng  # number of inequality constraints
    ncon = neqcon + nineqcon # number of constraints

    out['nb'] = nb
    out['nx'] = nx
    out['ng'] = ng
    out['ncon'] = ncon
    out['neqcon'] = neqcon
    out['nineqcon'] = nineqcon
    
    return out

import torch
#check violations of torch solutions
def check_PQbal(x_test, ppc, Pd_test, Qd_test, neg_Cg, nb, ng, nx, device ="cpu"):
    x_test, Pd_test, Qd_test, neg_Cg = x_test.to(device), Pd_test.to(device), \
                                       Qd_test.to(device), neg_Cg.to(device)
    # x_test: tensor (ntest, nx)
    # ppc: casedata
    # Pd_test: (ntest, nb) tensor
    # neg_Cg: (nb, ng) tensor, coefficient matrix turn Pg -> -Pg_bus
    Ybus_test, Yf_test, Yt_test = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])  # topo: Ybus = Variable('topo')
    Ybus_test = sparse.toarray(Ybus_test)  # to dense numpy (nb,nb)
    Ybusr_test = torch.tensor(np.real(Ybus_test)).float()  # Ybus real part, torch tensor
    Ybusi_test = torch.tensor(np.imag(Ybus_test)).float()  # imag part
    Ybusr_test, Ybusi_test = Ybusr_test.to(device), Ybusi_test.to(device)
    #
    Vr_test = x_test[:, 0:nb]  # rectangular
    Vi_test = x_test[:, nb:2 * nb]  # rectangular
    Pg_test = x_test[:, 2 * nb:2 * nb + ng]  # Pg
    Qg_test = x_test[:, 2 * nb + ng:nx]  # Qg
    res_Pbal = (Pg_test @ neg_Cg.t() +
                Vr_test * (Vr_test @ Ybusr_test.t()) -
                Vr_test * (Vi_test @ Ybusi_test.t()) +
                Vi_test * (Vi_test @ Ybusr_test.t()) +
                Vi_test * (Vr_test @ Ybusi_test.t()) + Pd_test)
    res_Qbal = (Qg_test @ neg_Cg.t() +
                Vi_test * (Vr_test @ Ybusr_test.t()) -
                Vi_test * (Vi_test @ Ybusi_test.t()) -
                Vr_test * (Vi_test @ Ybusr_test.t()) -
                Vr_test * (Vr_test @ Ybusi_test.t()) + Qd_test)
    return res_Pbal, res_Qbal



def check_bnds(x_test, Pgmax, Pgmin, Qgmax, Qgmin, Vm2max, Vm2min, nb, ng, nx, device="cpu"):
    # xtest: (ntest, nx) tensor
    # Pgmin/max: (ng, 1) tensor
    # Vmax/min: (nb,1) tensor
    x_test = x_test.to(device)
    Pgmax, Pgmin, Qgmax, Qgmin, Vm2max, Vm2min = Pgmax.to(device), Pgmin.to(device), \
                                                 Qgmax.to(device), Qgmin.to(device), \
                                                 Vm2max.to(device), Vm2min.to(device)
    Vr_test = x_test[:, 0:nb]  # rectangular
    Vi_test = x_test[:, nb:2 * nb]  # rectangular
    Pg_test = x_test[:, 2 * nb:2 * nb + ng]  # Pg
    Qg_test = x_test[:, 2 * nb + ng:nx]  # Qg
    #
    Pgmax_viol = torch.maximum(torch.zeros(1), Pg_test - Pgmax.t())  # tensor (ntest, ng) relu(Pg-Pgmax)
    n_Pgmax_viol = (Pgmax_viol > 0).sum(dim=-1)  # (ntest,)
    Pgmin_viol = torch.maximum(torch.zeros(1), Pgmin.t() - Pg_test)  # relu(Pgmin-Pg)
    n_Pgmin_viol = (Pgmin_viol > 0).sum(dim=-1)  #
    Qgmax_viol = torch.maximum(torch.zeros(1), Qg_test - Qgmax.t())  # tensor (ntest, ng) relu(Pg-Pgmax)
    n_Qgmax_viol = (Qgmax_viol > 0).sum(dim=-1)  #
    Qgmin_viol = torch.maximum(torch.zeros(1), Qgmin.t() - Qg_test)  # relu(Pgmin-Pg)
    n_Qgmin_viol = (Qgmin_viol > 0).sum(dim=-1)  #
    Vm2_test = Vr_test ** 2 + Vi_test ** 2
    Vm2max_viol = torch.maximum(torch.zeros(1), Vm2_test - Vm2max.t())  # tensor (ntest, nb)
    n_Vm2max_viol = (Vm2max_viol > 0).sum(dim=-1)
    Vm2min_viol = torch.maximum(torch.zeros(1), Vm2min.t() - Vm2_test)
    n_Vm2min_viol = (Vm2min_viol > 0).sum(dim=-1)
    res_bnds = torch.cat((Pgmax_viol, Pgmin_viol, Qgmax_viol, Qgmin_viol, Vm2max_viol, Vm2min_viol), dim=-1)
    #  output size: (ntest, nelement)
    return res_bnds, n_Pgmax_viol, n_Pgmin_viol, n_Qgmax_viol, n_Qgmin_viol, n_Vm2max_viol, n_Vm2min_viol

# deleted funcs
# def adjust_Pg_by_res(x_test, res_Pbal, res_Qbal, neg_Cg):
#     nb, ng = neg_Cg.size()
#     nx = 2*(nb+ng)
#     x_test[:,2*nb:2*nb+ng] = x_test[:, 2*nb:2*nb+ng] - res_Pbal@neg_Cg
#     x_test[:,2 * nb + ng:nx] = x_test[:, 2 * nb:2 * nb + ng] - res_Qbal @ neg_Cg
#     return x_test