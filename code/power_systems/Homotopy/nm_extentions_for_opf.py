import numpy as np
import torch
import torch.nn as nn

#
from pypower.api import runopf
from pypower.idx_bus import PD,QD, VM, VA, BUS_I
from pypower.idx_gen import PG,QG,VG, GEN_BUS
from pypower.api import runpf
from pypower.ppoption import ppoption
import copy

class BatchNorm(nn.Module):
    """
    batch norm block consistent with blocks interface
    """
    def __init__(
        self,
        insize
    ):
        """
        :param insize: (int) dimensionality of input
        """
        super().__init__()
        self.linear = nn.BatchNorm1d(num_features=insize)

    def forward(self, x):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize]), outsize=insize
        """
        x = self.linear(x)
        return x


def getlabels_acopf_pyp(case, Pds, Qds, mask=[]):
    """
    :INPUT ppc_case - pypower casedata,
    :INPUT Pd (nsample,nb)- vector of load values in p.u.
    :INPUT mask (nsample,) - vector of 0,1 denoting whether the sample is selected
    :OUTPUT x_label (nsample, nx) = [Pg, Qg, Vr, Vi] numpy array, p.u.
    """
    if len(mask) == 0:
        mask = np.ones(Pds.shape[0])
    ppc_case = copy.deepcopy(case)
    nb = ppc_case['bus'].shape[0]
    ng = ppc_case['gen'].shape[0]
    nx = 2*nb + 2*ng
    x_label = np.zeros((len(mask), nx))
    #
    baseMVA = ppc_case['baseMVA']
    ppopt = ppoption()
    ppopt['OUT_ALL'] = 0
    ppopt['VERBOSE'] = 0
    for i,flag in enumerate(mask):
        if flag:
            Pd, Qd = Pds[i]*baseMVA, Qds[i]*baseMVA
            ppc_case['bus'][:,PD] = Pd
            ppc_case['bus'][:,QD] = Qd
            pypower_result = runopf(ppc_case,ppopt)
            Pg_optim_pyp = pypower_result['gen'][:, PG]/baseMVA
            Qg_optim_pyp = pypower_result['gen'][:, QG]/baseMVA
            Vr_optim_pyp = pypower_result['bus'][:, VM] * np.cos(pypower_result['bus'][:, VA] / 180 * np.pi)
            Vi_optim_pyp = pypower_result['bus'][:, VM] * np.sin(pypower_result['bus'][:, VA] / 180 * np.pi)
            x_label[i] = np.concatenate((Vr_optim_pyp, Vi_optim_pyp, Pg_optim_pyp, Qg_optim_pyp))
            print("\r get supervised labels %d/%d/%d"%(i+1,mask.sum(),len(mask)), end = "")
    print("\n")
    return x_label



class Pypower_PF(nn.Module): #a block for self-supervision of voltages
    """
    pypower ac power flow solver, to be wrapped in neuromancer
    """

    def __init__(
            self,
            insize,
            case,
            perc=0.1,
            verbose = False
    ):
        """
        :param insize: (int) dimensionality of input
        :param perc: percentage of data to apply selfsup, generate mask accordingly
        :param outsize (int) output dimension
        """
        super().__init__()
        self.case = case
        self.ng = case['gen'].shape[0]
        self.nb = case['bus'].shape[0]
        self.perc = perc
        self.verbose = verbose
    def forward(self, x, Pds, Qds):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :Pds, Qds: torch tensor shape [batchsize, nb], load at every bus, pu
        :return mask: torch tensor (batchsize, 1), 0/1s denoting whether each sample is selected to supervise
        :return: x_label (torch.Tensor, shape=[batchsize, outsize]), outsize=insize
        """
        # detach x from torch
        nsample, nx = x.size()
        xnet = x.detach().numpy().reshape((nsample, nx))
        n_selfsup = round(self.perc*nsample) #min(round(self.perc*nsample)+1, nsample)
        mask = torch.zeros((nsample, 1))
        nsuccess = 0
        x_label = copy.deepcopy(xnet)
        if n_selfsup > 0:
            # generate mask
            idx = np.random.choice(nsample, n_selfsup, replace=False)
            mask[idx] = 1  # random mask
            # solve pf with Pg, Vg, Pd, Qd settings
            ppc_case = copy.deepcopy(self.case)
            nb, ng = self.nb, self.ng
            Pgnet = xnet[:,2*nb:2*nb+ng] #p.u.
            Qgnet = xnet[:,2*nb+ng:nx]
            Vrnet = xnet[:,0:nb]
            Vinet = xnet[:,nb:2*nb]
            Vmnet = np.abs(Vrnet+1j*Vinet)
            Vanet = np.angle(Vrnet+1j*Vinet)/np.pi*180 #change to degree
            baseMVA = ppc_case['baseMVA']
            ppopt = ppoption()
            if not self.verbose:
                ppopt['OUT_ALL'] = 0
                ppopt['VERBOSE'] = 0
            else:
                ppopt['OUT_ALL'] = 1
                ppopt['VERBOSE'] = 1
            #
            for i in idx:
                Pd, Qd = Pds[i] * baseMVA, Qds[i] * baseMVA
                Pg, Qg = Pgnet[i]*baseMVA, Qgnet[i]*baseMVA
                Vm, Va = Vmnet[i], Vanet[i]
                ppc_case['bus'][:, PD] = Pd
                ppc_case['bus'][:, QD] = Qd
                ppc_case['bus'][:, VM] = Vm
                ppc_case['bus'][:, VA] = Va
                ppc_case['gen'][:, PG] = Pg
                ppc_case['gen'][:, QG] = Qg
                pf_result, success = runpf(ppc_case, ppopt)
                if success:
                    Vr_optim_pyp = pf_result['bus'][:, VM] * np.cos(pf_result['bus'][:, VA] / 180 * np.pi)
                    Vi_optim_pyp = pf_result['bus'][:, VM] * np.sin(pf_result['bus'][:, VA] / 180 * np.pi)
                    Qg_optim_pyp = pf_result['gen'][:, QG]/baseMVA
                    x_label[i] = np.concatenate((Vr_optim_pyp, Vi_optim_pyp, Pgnet[i], Qg_optim_pyp)) #x_label[i] = [Vr, Vi, Pg, Qg]
                    nsuccess = nsuccess + 1
            print("self-supervision %d/%d" % (nsuccess, n_selfsup))
        # numpy array -> tensor
        x_label = torch.from_numpy(x_label) #(nsample, nx)
        return x_label, mask

    def forward_with_mask(self, x, Pds, Qds, mask=None, output_successflag=False):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :Pds, Qds: torch tensor shape [batchsize, nb], load at every bus, pu
        :mask: torch tensor (batchsize, 1), 0/1s denoting whether each sample is selected to supervise
        :return: x_label (torch.Tensor, shape=[batchsize, outsize]), outsize=insize
        """
        # detach x from torch
        nsample, nx = x.size()
        if mask == None:
            mask = torch.ones((nsample,1))
        xnet = x.detach().numpy().reshape((nsample, nx))
        n_selfsup = mask.sum()
        nsuccess = 0
        successflags = []
        x_label = copy.deepcopy(xnet)
        if n_selfsup > 0:
            # solve pf with Pg, Vg, Pd, Qd settings
            ppc_case = copy.deepcopy(self.case)
            nb, ng = self.nb, self.ng
            Pgnet = xnet[:,2*nb:2*nb+ng] #p.u.
            Qgnet = xnet[:,2*nb+ng:nx]
            Vrnet = xnet[:,0:nb]
            Vinet = xnet[:,nb:2*nb]
            Vmnet = np.abs(Vrnet+1j*Vinet)
            Vanet = np.angle(Vrnet+1j*Vinet)/np.pi*180 #change to degree
            baseMVA = ppc_case['baseMVA']
            ppopt = ppoption()
            if not self.verbose:
                ppopt['OUT_ALL'] = 0
                ppopt['VERBOSE'] = 0
            else:
                ppopt['OUT_ALL'] = 0
                ppopt['VERBOSE'] = 1
            #
            nsuccess = 0
            for i, flag in enumerate(mask):
                if flag:
                    Pd, Qd = Pds[i] * baseMVA, Qds[i] * baseMVA
                    Pg, Qg = Pgnet[i]*baseMVA, Qgnet[i]*baseMVA
                    Vm, Va = Vmnet[i], Vanet[i]
                    ppc_case['bus'][:, PD] = Pd
                    ppc_case['bus'][:, QD] = Qd
                    ppc_case['bus'][:, VM] = Vm
                    ppc_case['bus'][:, VA] = Va
                    ppc_case['gen'][:, PG] = Pg
                    ppc_case['gen'][:, QG] = Qg

                    pf_result, success = runpf(ppc_case, ppopt)
                    successflags.append(success)
                    if success:
                        Vr_optim_pyp = pf_result['bus'][:, VM] * np.cos(pf_result['bus'][:, VA] / 180 * np.pi)
                        Vi_optim_pyp = pf_result['bus'][:, VM] * np.sin(pf_result['bus'][:, VA] / 180 * np.pi)
                        Qg_optim_pyp = pf_result['gen'][:, QG] / baseMVA
                        x_label[i] = np.concatenate((Vr_optim_pyp, Vi_optim_pyp, Pgnet[i], Qg_optim_pyp))  # x_label[i] = [Vr, Vi, Pg, Qg]
                        nsuccess = nsuccess + 1
            print("self-supervision %d/%d" % (nsuccess, n_selfsup))
        # numpy array -> tensor
        x_label = torch.from_numpy(x_label) #(nsample, nx)
        if output_successflag:
            return x_label, successflags
        else:
            return x_label

from torch import nn
from neuromancer.component import Component
import torch
#class Wrapper(): # a map
# input keys e.g. ['x', 'Pds', 'Qds', 'selfsup-mask']
# output keys e.g. ['x_selfsup_label']
class Wrapper(Component):
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.func = func
        self.name = name

    def forward(self, data):
        inp = [data[k] for k in self.input_keys] #[xtensor, Pds, Qds, mas]
        out = self.func(*inp) #a tensor output from solver
        out_dict = {
            k: v for k, v in zip(
                self.output_keys,
                out if isinstance(out, tuple) else (out,)
            )
        }
        return out_dict

#
class param_scheduler:
    #schedule hyper paramter Q within the training cycle
    def __init__(self, Q0=0, stepsize = 100, mode='multiply', factor = 1, Qmax=1e10, Qmin=-1e10, openat=0, closeat=1e10, name='', verbose=True):
        self.stepsize = int(stepsize) #update paramter every stepsize epochs
        self.mode = mode #multiply, add
        self.factor = float(factor)
        #in each update step, do Qnew = Qold*factor, or Qold + factor, depending on mode
        self.name = name #name of parameter
        self.Qinit = float(Q0) #initial Q value
        self.Qmax = float(Qmax) #the maximum value of Q
        self.Qmin = float(Qmin) #the min value of Q
        self.nstep = 0 #how many update steps we have done so far
        self.verbose = verbose #if True, print at every update step
        #
        self.openat = openat # Q=0 before certain step number, Q tuning starts from Qinit after that step number
        self.closeat = closeat #Q=0 after certain step
        #
        self.Qcur = float(0.0) if self.openat>0 else self.Qinit #the current Q value
        if self.verbose:
            print("-- %s scheduler: init %f, %s by %f every %d epochs, value range (%d, %d)"%(self.name,
                                                                    self.Qcur,
                                                                    self.mode, self.factor,
                                                                    self.stepsize,
                                                                    self.Qmin, self.Qmax))
    def step(self, iterCount):
        nstep = iterCount//self.stepsize
        is_active = (nstep>=self.openat and nstep<self.closeat)
        if nstep>self.nstep:
            if not is_active:
                self.Qcur = float(0.0)
                if self.verbose:
                    print("Inactive tuning, %s=%f" % (self.name, self.Qcur))
            else:
                if self.nstep<self.openat:
                    # start gate,
                    self.Qcur = self.Qinit
                    if self.verbose:
                        print("start %s at %f" % (self.name, self.Qcur))
                else:
                    if self.mode == 'multiply':
                        self.Qcur = self.Qcur*self.factor
                    elif self.mode == 'add':
                        self.Qcur = self.Qcur + self.factor
                    #
                    if self.Qcur>self.Qmax:
                        self.Qcur = self.Qmax
                    elif self.Qcur<self.Qmin:
                        self.Qcur = self.Qmin
                    #
                    if self.verbose:
                        print("updated %s weight to %f, factor=%f"%(self.name,
                                                                    self.Qcur,
                                                                    self.factor))
            self.nstep = nstep
        return self.Qcur


def NN_initializer(fblock, method = 'xavier'):
    for lin in fblock.linear:
        if method == 'normal':
            nn.init.normal_(lin.weight)
        elif method == 'xavier':
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)

def get_tag_general(args):
    #generate acopf tag
    tag = ''
    if args.loss == 'augmented_lagrange':
        tag = tag + 'Lag_'
    if args.add_proj:
        tag = tag + 'Proj' + args.proj_option + '_'

    if args.use_paramscheduler:
        if args.homo_delta < 0.01:
            tag = tag + 'HomoD%.3f_' % (args.homo_delta)
        else:
            tag = tag + 'HomoD%.2f_' % (args.homo_delta)
        tag = tag + ('%s_'%args.obj_homo_option if args.obj_homo_option!='' else '')
        tag = tag + ('%s_'%args.coneq_homo_option if args.coneq_homo_option!='' else '')
        if args.coneq_homo_option == 'SaS' and (args.coneq_eH!=0.01 or args.coneq_eL!=0.0):
            tag = tag + 'eH%.2feL%.5f_'%(args.coneq_eH, args.coneq_eL)
        tag = tag + ('%s_'%args.conineq_homo_option if args.conineq_homo_option!='' else '')
        if args.conineq_homo_option=='ShrinkBnds':
            tag = tag + 'e%.2f_'%args.conineq_e
    else:
        tag = tag + 'vanilla_'

    if args.use_paramscheduler and args.hyperstep!=100:
        tag = tag + 'OuterStep%d_'%args.hyperstep
    #
    if args.Q != 1:
        tag = tag + 'Q%.1f'%args.Q
    #
    if args.data_seed!=408 or args.NN_seed!=408:
        tag = tag + 'dseed%dnnseed%d_'%(args.data_seed, args.NN_seed)
    tag = tag + 'L%d'%args.n_layers
    tag = tag + 'W%d'%args.nx_hidden
    # constraint weight
    tag = tag + '_Qeq%d'%args.Q_coneq + '_Qineq%d' % args.Q_conineq
    return tag

def get_tag_acopf(args):
    #generate acopf tag
    #default hyper step is case9: 100
    tag = ''
    if args.loss == 'augmented_lagrange':
        tag = tag + 'Lag_'
    if args.add_proj:
        tag = tag + 'Proj' + args.proj_option + '_'

    if args.use_paramscheduler:
        if args.homo_delta < 0.01:
            tag = tag + '%sDelta%.3f_'%(args.coneq_homo_option, args.homo_delta)
        else:
            tag = tag + '%sDelta%.2f_'%(args.coneq_homo_option, args.homo_delta)

        if args.coneq_homo_option == 'SaS' and (args.con_res_H!=0.1 or args.con_res_L!=0.0001):
            tag = tag + 'H%.2fL%.5f'%(args.con_res_H, args.con_res_L)
        if args.add_bnds_homo:
            tag = tag + 'HomoBnds'
            if args.Pgmax_homofactor!=1:
                tag = tag + 'Pg%d'%args.Pgmax_homofactor
            tag = tag + "_"
    else:
        tag = tag + 'vanilla_'

    if args.use_paramscheduler and args.hyperstep!=100:
        tag = tag + 'OuterStep%d_'%args.hyperstep
    #
    if args.Q != 1:
        tag = tag + 'Q%.1f'%args.Q
    #
    if args.data_seed!=408 or args.NN_seed!=408:
        tag = tag + 'dseed%dnnseed%d_'%(args.data_seed, args.NN_seed)
    tag = tag + 'L%d'%args.n_layers
    tag = tag + ('W%d'%args.nx_hidden if args.nx_hidden!=200 else '')
    #
    tag = tag + '_%dsamples'%args.nsamples
    if args.lf_L!=0.75 or args.lf_H!=1.5:
        tag = tag + 'lfL%.2fH%.2f'%(args.lf_L,args.lf_H)
    #
    # PUll up Pg and Vg
    if args.add_pullup and (args.Q_Pgup>0):
        #PgUp, VgUp, or PgVgUp
        tag = tag + '_'+('Pg' if args.Q_Pgup > 0 else '')+('tol%.2f'%args.tol_Pgup if args.tol_Pgup>0 else '') + 'Up'
    # equality constraint weight
    if args.Q_coneq != 1e5:
        tag = tag + '_Qeq%d'%args.Q_coneq
    if args.Q_conineq != 1e6:
        tag = tag + '_Qineq%d' % args.Q_conineq
    return tag

def get_tag(args):
    #no longer used
    #generate acopf tag
    #default hyper step is case9: 100
    tag = ''
    if args.loss == 'augmented_lagrange':
        tag = tag + 'Lag_'
    if args.add_proj:
        tag = tag + 'Proj' + args.proj_option + '_'
    keys = []
    if args.add_coneq_homo:
        tag = tag + '%sDelta%.2f_'%(args.coneq_homo_option, args.homo_delta)
        if args.coneq_homo_option == 'SaS' and (args.con_res_H!=0.1 or args.con_res_L!=0.0001):
            tag = tag + 'H%.2fL%.5f'%(args.con_res_H, args.con_res_L)
    keys.append(args.add_coneq_homo)
    if args.add_bnds_homo:
        assert args.add_coneq_homo, "add_coneq_homo not True, homo bnds cannot work alone"
        tag = tag + 'HomoBnds'
        if args.Pgmax_homofactor!=1:
            tag = tag + 'Pg%d'%args.Pgmax_homofactor
        tag = tag + "_"
    #
    if args.add_sup_homo:
        tag = tag + 'HomoSup%dDelta%.2f_'%(int(args.perc_sup*100), args.homo_delta)
    keys.append(args.add_sup_homo)
    # if args.add_weight_homo:
    #     tag = tag + '%sDelta%f_'%(args.weight_homo_option, args.homo_delta)
    if args.add_sup:
        tag = tag + 'Sup%dQ%d_'%(int(args.perc_sup*100), int(args.Q_sup))
    keys.append(args.add_sup)
    if args.add_selfsup:
        tag = tag + 'SS%dQss%d_'%(int(args.perc_sup*100), int(args.Q_selfsup))
    keys.append(args.add_selfsup)
    if any(keys):
        assert args.use_paramscheduler, "use_paramscheduler is off, cannot use hyper tuning"

    #
    is_vanilla = not any(keys)
    if is_vanilla:
        if args.use_paramscheduler:
            tag = tag + 'vanillaO_'
        else:
            tag = tag + 'vanilla_'

    if args.use_paramscheduler and args.hyperstep!=100:
        tag = tag + 'OuterStep%d_'%args.hyperstep
    #
    if args.Q != 1:
        tag = tag + 'Q%.1f'%args.Q
    #
    if args.data_seed!=408 or args.NN_seed!=408:
        tag = tag + 'dseed%dnnseed%d_'%(args.data_seed, args.NN_seed)
    tag = tag + 'L%d'%args.n_layers
    tag = tag + ('W%d'%args.nx_hidden if args.nx_hidden!=200 else '')
    #
    tag = tag + '_%dsamples'%args.nsamples
    if args.lf_L!=0.75 or args.lf_H!=1.5:
        tag = tag + 'lfL%.2fH%.2f'%(args.lf_L,args.lf_H)
    #
    # PUll up Pg and Vg
    if args.add_pullup and (args.Q_Vup + args.Q_Pgup>0):
        #PgUp, VgUp, or PgVgUp
        tag = tag + '_'+('Pg' if args.Q_Pgup > 0 else '')+('tol%.2f'%args.tol_Pgup if args.tol_Pgup>0 else '')+('Vg%d'%(args.Q_Vup) if args.Q_Vup != 1 else '')+'Up'
    # equality constraint weight
    if args.Q_coneq != 1e5:
        tag = tag + '_Qeq%d'%args.Q_coneq
    if args.Q_conineq != 1e6:
        tag = tag + '_Qineq%d' % args.Q_conineq
    return tag

import matplotlib.pyplot as plt
def plot_lines(x2plot, y2plot,
               fig_file_name=None,
               xlabel2plot='', ylabel2plot='',
               legend2plot=None,
               title = '',
               figsize = (6,4)):
    # x2plt: 1d, (npoint,)
    # y2plot: 2d array (ncurve, npoint)
    plt.figure(figsize=figsize)
    for i in range(len(y2plot)):
        plt.plot(x2plot, y2plot[i].tolist())
    plt.xlabel(xlabel2plot)
    plt.ylabel(ylabel2plot)
    if legend2plot != None:
        plt.legend(legend2plot)
    if not fig_file_name:
        fig_file_name = 'unnamed_fig.png'
    plt.title(title)
    plt.savefig(fig_file_name)
    print("figure saved at", fig_file_name)
    #plt.show()

def plot_bars(names, values,
             fig_file_name=None,
         xlabel2plot='', ylabel2plot='',
         legend2plot=None,
              title = '',
              figsize = (6,7)):
    #plot multiple bars
    #names: 1d names (npoints,) [name1, name2, ...]
    #values: 2d values [nbar, npoints]
    nbar = len(values)
    barwidth = 0.2
    K = np.ceil(nbar*barwidth)
    plt.figure(figsize=figsize)
    X_axis = np.arange(len(names))*K
    for i in range(nbar):
        plt.bar(X_axis + barwidth*i - 0.5, values[i].tolist(), width=barwidth)
    plt.xticks(X_axis, names)
    #
    plt.xlabel(xlabel2plot)
    plt.ylabel(ylabel2plot)
    if legend2plot != None:
        plt.legend(legend2plot)
    if not fig_file_name:
        fig_file_name = 'unnamed_fig.png'
    plt.title(title)
    plt.savefig(fig_file_name)
    print("figure saved at", fig_file_name)
