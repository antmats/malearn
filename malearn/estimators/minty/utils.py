import time

import numpy as np
from gurobipy import *


def rule_assignments(X, S):
    """ 
    Computes the assignments of observations in X to rules in S
    
    args: 

    X : numpy array of shape (m x d) where m is the number of observations and d the number of features
    S : numpy array of shape (d x k) where k is the number of rules and d the number of features

    Note: 
    - Expects X to have an intercept column at [:,0] 
    """

    a = 1*(np.dot(X, S)>0)

    return a  


def missingness_reliance(X, M, S):
    """
    Returns the reliance in S on columns of X with missing values in M 

    The returned value is the average number of rules that have no observed value across both rules and observations
    
    args: 
    X : features of shape (n x d)
    M : missingness mask of shape (n x d)
    S : rule matrix of shape (d x k)
    
    """
    n = X.shape[0]
    k = S.shape[1]
 
    if k == 0:
        return np.zeros((n, 1))
    
    rho = np.zeros((n, k))
    for r in range(k):
        Sr = S[:, r] # Sr is the rth rule in S (d, ) across all d features -> (1, d)
        
        rho[:,r] = (1-np.max((Sr>0) & (1-M) & (X==1), axis=1)) * np.max((Sr>0) & M, axis=1)  
       
    return rho


def column_gen_ilp_(X, R, r, r_pos, relaxed=True, gamma=0.1, lambda_0=0.01, lambda_1=0.01, 
                   optimality_tol=1e-6, feasibility_tol=1e-6):
    """
    Solves the column generation problem with ILP solver for a single sign of the loss.
    
    X (numpy array, n x d) : Binary input features (with all-one 1st column)
    R (numpy array, n x d) : Binary missingness mask
    r (numpy array, n x 1) : Residual for current model
    r_pos (bool)           : True if solving with positive loss, False with negative
    relaxed (bool)         : If TRUE then the model is trained with the relaxed 
                             objective and the constraint are 'relaxed', otherwise 
                             without is has the original objective and constraint
    gamma (float > 0)   : Slack variable for the hinge loss
    lambda_0 (float > 0): Regularization per rule
    lambda_1 (float > 0): Regularization per literal in each rule
    optimality_tol (float>0)  : Optimality tolerance of the ILP solver
    feasibility_tol (float>0) : Feasibility tolerance of the ILP solver
    """

    # Create a gurobi model
    m = Model()
    # Silence Gurobi output
    m.setParam('OutputFlag', False)
    m.setParam('OptimalityTol', optimality_tol)
    m.setParam('IntFeasTol', feasibility_tol)

    # Initialization
    n = X.shape[0]
    d = X.shape[1]

    X = X.astype(int)

    # minimize for a, z with positive  ub=1, lb= 0,
    z = m.addVars(d, vtype=GRB.BINARY,
                  name="representation")  # represents if feature j is selected in a disjunction
    a = m.addVars(n, vtype=GRB.BINARY, name="assignment")  # a_ik value taken by A_k in instance i

    rho = m.addVars(n, vtype=GRB.BINARY, name="slack variable")

    sign = 1. if r_pos else -1.

    loss = sign*sum([0.5 / n * r[i, 0] * a[i] for i in range(n)])
    size_reg = lambda_0 + lambda_1 * (sum([z[j] for j in range(d)]))
    miss_reg = gamma / n * sum([rho[i] for i in range(n)])

    obj = loss + size_reg
    if relaxed and gamma > 0:
        obj += miss_reg
    else:
        obj += 0


    # objective
    m.setObjective(obj, GRB.MINIMIZE)

    # constraints
    qs = []
    for i in range(n):
        qi = m.addVars(d, lb=0, ub=1)  # generates d variables
        qs.append(qi)

        m.addConstrs(qi[j] == X[i, j] * z[j] for j in range(d))

        # constraints to get disjunctions
        m.addGenConstrMax(a[i], [qi[j] for j in range(d)])

    #Constraint for relaxed version
    if relaxed:   
        for i in range(n):
            qt = m.addVars(d, lb=0, ub=1)  # generates d variables
            qo = m.addVars(d+1, lb=0, ub=1)  # generates d variables
            qs = m.addVars(d, lb=0, ub=1)  # generates d variables

            # @TODO: Need to count cases like (0,0,0,NaN) as relying on missingness because otherwise
            #        the model implicitly assumes that the NaN is a 0
            for j in range(d):
                m.addConstrs(qt[j] == (1 - R[i, j]) for j in range(d))
                m.addConstrs(qo[j] == (qt[j] * z[j] * X[i, j]) for j in range(d))

                m.addConstrs(qs[j] == (z[j] * R[i, j]) for j in range(d))

            qr = m.addVar(lb=0, ub=1)  # generates 1 variable
            qp = m.addVar(lb=0, ub=1)  # generates 1 variable

            m.addGenConstrMax(qp, [qs[j] for j in range(d)])

            m.addConstr(qo[d] == (1-qp))

            m.addGenConstrMax(qr, [qo[j] for j in range(d+1)])

            m.addConstr(rho[i] == (1-qr))
    else:
        # Constraints for base version
        for i in range(n):
            # Update with max over j
            m.addConstr(sum((1 - R[i, j]) * z[j] for j in range(d)) >= 1)


    # Determines whether dual reductions are performed during the optimization process.
    m.Params.DualReductions = 0
    # Solve model
    m.optimize()

    #save values from a and z from all iterations
    a_ = np.array([a[i].x for i in range(n)])
    z_ = np.array([z[j].x for j in range(d)])
    rho_ = np.array([rho[i].x for i in range(n)])
    
    return m.objVal, a_, z_, rho_


def column_gen_ilp(X, M, r, relaxed=True, gamma=0.1, lambda_0=0.01, lambda_1=0.01, 
                   optimality_tol=1e-6, feasibility_tol=1e-6):
    """    
    Solves the column generation problem with ILP solver.
    
    X (numpy array, n x d) : Binary input features (with all-one 1st column)
    R (numpy array, n x d) : Binary missingness mask
    r (numpy array, n x 1) : Residual for current model
    relaxed (bool)         : If TRUE then the model is trained with the relaxed 
                             objective and the constraint are 'relaxed', otherwise 
                             without is has the original objective and constraint
    gamma (float > 0)   : Slack variable for the hinge loss
    lambda_0 (float > 0): Regularization per rule
    lambda_1 (float > 0): Regularization per literal in each rule
    optimality_tol (float>0)  : Optimality tolerance of the ILP solver
    feasibility_tol (float>0) : Feasibility tolerance of the ILP solver
    """
    
    o_p, a_p, z_p, rho_p = column_gen_ilp_(X, M, r, True, relaxed=relaxed, gamma=gamma, 
                                      lambda_0=lambda_0, lambda_1=lambda_1, optimality_tol=optimality_tol)
    o_m, a_m, z_m, rho_m = column_gen_ilp_(X, M, r, False, relaxed=relaxed, gamma=gamma, 
                                      lambda_0=lambda_0, lambda_1=lambda_1, optimality_tol=optimality_tol)

    # check if both optimal
    best_obj = np.inf
    best_rule = None
    best_rho = None
    if o_p < 0 or o_m < 0:  
        if o_m < o_p:
            z_ = np.abs(np.round(z_m))
            best_obj = o_m
            best_rho = rho_m
        else:
            z_ = np.abs(np.round(z_p))
            best_obj = o_p                   
            best_rho = rho_p
                                             
        best_rule = z_.reshape(-1, 1)                                    

    return best_rule, best_obj, best_rho


def evaluate_rules_(X, M, Z, r, lambda_0=0, lambda_1=0, gamma=0):
    """    
    Computes the column generation objective for the rules in Z
    """
        
    n = X.shape[0]
    k = Z.shape[1]

    # Compute assignments
    t0 = time.time()
    a = rule_assignments(X, Z)
    #print('Assignments: %.2fs' % (time.time() - t0))
    
    # Compute correlation between residual and rule assignments
    C = np.dot(r.T, a).ravel()

    # Compute loss part of objective
    loss = 0.5/n*C

    # Compute size regularization
    size_reg = lambda_0 + lambda_1*Z.sum(axis=0)

    # Compute missingness reliance
    if gamma>0:
        rel = missingness_reliance(X, M, Z)
        miss_reg = gamma*rel.mean(axis=0)
    else:
        rel = np.zeros((n, k))
        miss_reg = 0

    # Compute objective
    obj_p = loss + size_reg + miss_reg
    obj_m = -loss + size_reg + miss_reg
    objs = np.vstack([obj_p, obj_m])
    obj = np.min(objs, axis=0)

    return obj, rel


def expand_rules_(Z):
    """    
    Proposes new rules to add to the selection in Z by adding all combinations of one literals with every existing rule
    """
    
    d = Z.shape[0]
    k = Z.shape[1]

    # Find all rules where a "1" has been added in any position
    Zp = (Z.reshape(d,-1,1)+np.eye(d,d).reshape(d,1,d)).reshape(d,d*k)

    # Keep valid rules (no "2")
    Zp = Zp[:,Zp.max(axis=0)==1]

    # Remove duplicates
    Zp = np.unique(Zp, axis=1)

    return Zp


def column_gen_beam(X, M, r, gamma=0.1, lambda_0=0.01, lambda_1=0.01, beam_width=10, beam_depth=10):
    """
    Solves the column generation problem with beam search solver.
    
    X (numpy array, n x d) : Binary input features (with all-one 1st column)
    R (numpy array, n x d) : Binary missingness mask
    r (numpy array, n x 1) : Residual for current model
    gamma (float > 0)   : Slack variable for the hinge loss
    lambda_0 (float > 0): Regularization per rule
    lambda_1 (float > 0): Regularization per literal in each rule
    beam_width (float>0) : Beam width of the beam search solver
    beam_depth (float>0) : Beam depth of the beam search solver
    """
    
    # Dimensions
    m = X.shape[0]
    d = X.shape[1]
    beam_depth = int(beam_depth)
    beam_width = int(beam_width)

    # Maintain best-ever rule
    best_rule = None
    best_obj = np.inf
    best_rho = None

    # Candidate rules of size d x k
    Z = np.eye(d)
    k = Z.shape[1]

    # Perform beam search

    for i in range(beam_depth):
        # Evaluate rules
        t0 = time.time()
        obj, rhos = evaluate_rules_(X, M, Z, r, lambda_0=lambda_0, lambda_1=lambda_1, gamma=gamma)
        #print('Evaluate rules: %.2fs' % (time.time() - t0))

        # Find minimizer of objective
        I_min = np.argsort(obj)

        if obj[I_min[0]] < best_obj:
            best_obj = obj[I_min[0]]
            best_rule = Z[:,I_min[0]].reshape(-1,1)
            best_rho = rhos[:,I_min[0]].mean()

        # Select top rules
        ZB = Z[:,I_min[:beam_width]]

        # Expand the beam
        Z = expand_rules_(ZB)
        
        # If no more rules to add
        if Z.shape[1] < 1:
            break
        
    return best_rule, best_obj, best_rho
