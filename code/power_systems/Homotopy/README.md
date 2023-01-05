# Homotopy optimization on constrained DL
Homotopy heuristics are developed for constrained deep learning models, by transforming objective function and constraints.

## General homotopy heuristics
Let fobj denote the original objective function, h=0 denote the eq constraints, g \leq 0 denote the ineq constraints, and lambda_H denote the homotopy parameter that increases from 0 to 1 during the homotopy process. We first list some general homotopy heuristics that are applicable to all domains.

### Convexify Objective (CvxObj)

during the homotopy process, we start with minimizing a convex objective function fcvx, and gradually convert to the original fobj, 

    f_homo = lambda_H*fobj + (1-lambda_H)*fcvx

### Split and Shrink (SaS) for equality constraints h=0:
Any eq constraint h = 0 can be split into two constraints

    0 \leq h \leq 0
The two ineq constraints can be relaxed to 

    -e \leq h \leq e
where e is a purturbation. 
During homotpy process, we can gradually reduce the relaxation by
            
    -(1-lambda_H)*e_H -e_L \leq h \leq (1-lambda_H)*e_H + e_L
so that it transforms from an easier constraint to the original one
### For inequality constraints g \leq 0
#### 1) Shrink bounds (ShrinkBnds): 
g \leq 0 can be relaxed to

    g \leq e
During the homotopy process, we can gradually decrease the purturbation by
                  
    g \leq (1-lambda_H) * e 
As e decreases, it gets closer to the original constraints
#### 2) Grow penalty (GrowPen):
when using augmented loss functiogn, we can gradually increase the penalty strength of the violation by increasing the penalty term weights:

    Loss = fobj + lambda_H*w*||h||^2

## Domain specific homotopy heuristics

Domain knowledge can be used to design the tranformation of eq constraints in power grid optimization problems.
(In this work, we test on the AC optimal power flow problem (ACOPF) problem.)
The eq constraints represents nonlinear network equations that describes the system power balance at every node
 
### 2) Load-stepping: 
When we gradually increase the load demand from 0 to the original value, a trivial solution to the first homotopy problem can be easily available 
based on domain knowledge: all power supply, load demand, and bus voltages are zero.
we can add a warmhomoloss to guide the initial update of the NN, towards the trivial solution
### 2) Tx-stepping: 
We gradually change the branches from short branches to the original branch

## File structure in this repo
- trainer_new.py -- neuromancer trainer (modified for stepLR) 
- nm_extention_for_opf.py -- some util functions (e.g. NN_initializer, get_method_name, etc)

- mpNLP_lincon_nm_rnd_homotopy.py -- code for running one deep learning model to approximate an nonlinear programming problem (nonconvex obj, ineq constraints)
- mp_NLP_meta.py -- script for training a set of nonlinear programming (NLP) models

- mpACOPF_nm_homotopy.py -- code for running (train/test) one deep learning model that approximates acopf problem
- acopfdisp.py -- contains some parser for acopf problem, and some post-processing funcs
- mpACOPF_meta.py -- script for training a set of acopf models with different hyperparameters

- (pnnl gpu) dispatch_acopf.py -- dispatch script for running on pnnl gpu 
- (pnnl gpu) dispatch_NLP.py -- dispatch script for running on pnnl gpu
- (pnnl gpu) environment_new.yml -- environment file for creating the python env on pnnl gpu, copy it under neuromancer repo and use to to create conda env
