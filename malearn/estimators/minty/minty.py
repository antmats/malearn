import numpy as np
import pandas as pd
from gurobipy import *

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.utils.validation import check_X_y
from sklearn.base import is_classifier, BaseEstimator
from sklearn.utils.multiclass import unique_labels

from .utils import *
from ..base import ClassifierMixin, RegressorMixin

LR_SOLVER = 'liblinear'


class BaseMinty(BaseEstimator):
    def __init__(self, optimizer='beam', max_rules=7, lambda_0=0.01, lambda_1=0.01, gamma=0.1, 
                 reg_refit=False, relaxed=True, optimality_tol=1e-6, feasibility_tol=1e-6,
                 beam_width=10, beam_depth=10, reg_rho=False, silent=False):
        """
        Args:
            optimizer (str)     : Which optimizer to use -- "ilp" or "beam" are valid
            max_rules (int)     : Maximum number of rules to learn
            lambda_0 (float > 0): Regularization per rule
            lambda_1 (float > 0): Regularization per literal in each rule
            reg_refit (bool)    : Whether to regularize the refit of the model at the end of
                                  training (with lambda_0) as parameter
            gamma (float > 0)   : Slack variable for the hinge loss
            relaxed (bool)      : If TRUE then the model is trained with the relaxed 
                                    objective and the constraint are 'relaxed', otherwise 
                                    without is has the original objective and constraint
            optimality_tol (float>0)  : Optimality tolerance of the ILP solver
            feasibility_tol (float>0) : Feasibility tolerance of the ILP solver
            beam_width (float>0) : Beam width of the beam search solver
            beam_depth (float>0) : Beam depth of the beam search solver
            reg_rho (bool)      : Whether to regularize the refit of the model at the end of
            silent (bool) : Suppresses all output if true
            """
        
        self.optimizer = optimizer
        self.max_rules = max_rules
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.gamma = gamma
        self.reg_refit = reg_refit
        self.relaxed = relaxed
        self.optimality_tol = optimality_tol
        self.feasibility_tol = feasibility_tol
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.reg_rho = reg_rho
        self.silent = silent
        
        self.fitted = False

    def assignments(self, X, S=None):
        """ Expects X to have an intercept column at [:,0] """

        if S is None:
            S = self.S

        a = (np.repeat(X[:, :, np.newaxis], S.shape[1], 2) * S).max(axis=1)

        return a

    def predict(self, X):
        self.ensure_fitted()
        
        # Compute rule assignments
        X_m = np.c_[np.ones(X.shape[0]), X]
        X_m[np.isnan(X_m)] = 0

        a = rule_assignments(X_m, self.S)

        # self.model does not expect intercept column
        if a.shape[1] < 2:
            yp = self.model.predict(a[:, 0:])  # DummyRegressor used
        else:
            yp = self.model.predict(a[:, 1:])

        return yp

    def rules(self, form='df', with_intercept=False, columns=None, S=None):
        """
        Note: 
            Counts the intercept as column 0 and removes that from rules automatically. 
            The argument "columns" should _not_ have a label for the intercept column. 
        """

        if S is None:
            S = self.S

        d = S.shape[0]
        if columns is None:
            columns = [str(j) for j in range(d)]

        if not with_intercept:
            S = S[:, 1:]

        if form == 'df':
            return pd.DataFrame(S.T)
        elif form == 'list':
            r = []
            for i in range(S.shape[1]):
                r.append(np.where(S[:, i])[0])
            return r
        elif form == 'str':
            r = []
            for i in range(S.shape[1]):
                r.append(' or '.join([columns[j-1] for j in np.where(S[:, i])[0]]))
            return r
        else:
            return S.T
        
    def ensure_fitted(self):
        """ Checks whether the model has been fit """
        if not self.fitted:
            raise Exception('Model has not been fitted.')
        return
        
    def compute_missingness_reliance(self, X, M, summarize=True, return_value="max"):
        """
        Returns the reliance on columns with missing values (zero imputation) in the predictions for X. 
        
        The first returned value is the average number of rules that have no observed value across both rules and observations. 
        The second return value is the average number of observations with at least one rule unobserved. 
        
        """

        assert return_value in ["max", "mean", None]
    
        self.ensure_fitted()
        
        S = self.S[1:,1:]

        M = np.isnan(X)
        
        rho = missingness_reliance(X, M, S)
        
        if summarize:
            rho_max = rho.max(axis=1).mean()
            rho_mean = rho.mean()

            if return_value == "max":
                return rho_max
            elif return_value == "mean":
                return rho_mean
            else:
                return rho_mean, rho_max
        else: 
            return rho
    
    def describe(self, columns=None, R=None, coef=None, display_zeros=False):
        """
        Note: 
            Counts the intercept as column 0 and removes that from rules automatically. 
            The argument "columns" should _not_ have a label for the intercept column. 
        """
        
        if R is None:
            R = self.rules(form='str', with_intercept=False, columns=columns)
        if coef is None:
            coef = self.coef_
            
        # Added this here because coef is stored as 2D from fit
        coef = coef.ravel()

        int_str = 'intercept'
        max_len = np.max([len(r) for r in R] + [len(int_str)])

        def rpad(s, l):
            return s + ' ' * (l - len(s))

        lines = []
        for i in range(len(R)):
            if display_zeros or np.abs(coef[i])>0:
                lines.append('%s %+.2f' % (rpad(R[i], max_len + 1), coef[i]))
        lines.append('%s %+.2f' % (rpad(int_str, max_len + 1), self.intercept_))

        return '\n'.join(lines)

    def describe_instance(self, X, columns=None, only_active_rules=True):
        """
        Note: 
            Counts the intercept as column 0 and removes that from rules automatically. 
            The argument "columns" should _not_ have a label for the intercept column. 
        """

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        d = X.shape[1]
        if n > 1:
            raise Exception('Can only describe the prediction for a single instance at a time')


        R = np.isnan(X).astype(int)
        R = np.c_[np.ones(n), R]
        X = np.nan_to_num(np.c_[np.ones(n), X], nan=0)

        S = self.S[:, 1:]
        if only_active_rules:
            S_x = S * (1 - R).T * X.T
        else:
            S_x = S * (1 - R).T

        k = S_x.shape[1]
        used = np.zeros(k)
        coef_ = []
        S_u = []
        for i in range(k):
            if used[i] or S_x[:, i].sum() < 1:
                continue
            I = np.where(np.all(S_x == S_x[:, i:(i + 1)], axis=0))
            used[I] = 1
            S_u.append(S_x[:, i])
            coef = self.coef_.ravel()[I].sum()
            coef_.append(coef)
        coef_ = np.array(coef_)

        if len(S_u) > 0:
            S_u = np.c_[np.array([1] + [0] * d), np.array(S_u).T]
            R = self.rules(form='str', with_intercept=False, columns=columns, S=S_u)
        else:
            R = []

        return self.describe(columns=columns, R=R, coef=coef_)
    
    def _reg_strength(self, X, M, S):
        """
        Computes the per-rule regularization strength
        """
        nf = S.sum(axis=0)
        lambda_k = self.lambda_0 + (nf[1:] * self.lambda_1).reshape(1, -1)
        if self.reg_rho: 
            rho = missingness_reliance(X[:,1:], M[:, 1:], S[1:,1:]).mean(axis=0)
            lambda_k += self.gamma*rho.reshape(1, -1)
        
        return lambda_k
    
    def _reg_weights(self, X, M, S, alpha=None):
        """
        Computes the weights for re-weighted LASSO
        """
        n = X.shape[0]
        
        if alpha is None:
            alpha = self.lambda_0

        lambda_k = self._reg_strength(X, M, S)
        
        w = lambda_k.repeat(n, 0) / alpha
        
        return 1./w, alpha
    
    def _fit_beta(self, X_m, R_m, Y, S, regularize=True, reweight_coef=False):
    
        # Initiate a by extending X with a new axis and muliply with S
        a = rule_assignments(X_m, S)
        # Fit teh best model under the current rule set
        if a.shape[1] < 2:
            model = None
            #r = (Y - Y.mean()) 
            # Class-based handling (e.g., probabilities for binary classification)
            r = (Y.ravel() == pd.Series(Y.ravel()).mode().iloc[0]).astype(int) - 0.5
        else:
            if regularize:
                w, alpha = self._reg_weights(X_m, R_m, S)
                aw = a[:, 1:] * w
                if is_classifier(self): 
                    model = LogisticRegression(fit_intercept=True, penalty='l1', C=1./alpha, solver=LR_SOLVER)
                    model.fit(aw, Y.ravel())
                    r = Y - model.predict_proba(aw)[:,1].reshape(-1, 1)
                    if reweight_coef:
                        model.coef_ = model.coef_*w[0,:]
                else:
                    model = Lasso(fit_intercept=True, alpha=alpha)
                    model.fit(aw, Y.ravel())
                    r = Y - model.predict(aw).reshape(-1, 1)
                    if reweight_coef:
                        model.coef_ = model.coef_*w[0:1,:]
            else:
                if is_classifier(self):           
                    model = LogisticRegression(C=1e8, penalty='l2', fit_intercept=True, solver=LR_SOLVER)
                    model.fit(a[:,1:], Y.ravel())      
                    r = Y - model.predict_proba(a[:,1:])[:,1].reshape(-1, 1)
                else:
                    model = LinearRegression(fit_intercept=True).fit(a[:, 1:], Y)
                    model.fit(a[:,1:], Y)
                    r = Y - model.predict(a[:,1:]).reshape(-1, 1)

                                
        return model, r

    def fit(self, X, Y):
        '''
        X: input (including nans to indicate missingness)
        Y: output

        The algorithm alternates between solving the restricted log-likelihood problem (4)
        and the searching for new columns by solving (9) for both signs

        The algorithm terminates with a certifcate that problem (4) is solved optimally if
        the optimal values of problem (9) is non-negative for both signs
        '''
        X, Y = check_X_y(X, Y, force_all_finite="allow-nan")

        # Initialization
        n = X.shape[0]
        d = X.shape[1]
        X_m = X.copy()
        
        # Reshape Y to be 2D
        Y = Y.reshape(-1, 1)

        # Add column of 1 to X and R for intercept term
        X_m = np.c_[np.ones(n), X_m]
        #X_m = X_m.astype(float)
        R_m = np.isnan(X_m)
        #X_m = X_m.astype(float)
        X_m[np.isnan(X_m)] = 0  # Test element-wise for NaN and return result as a boolean array

        # Initialization of rule definitions (start with only intercept column)
        S = np.array([[1] + [0] * (d)]).T
        self.S = S
        
        # Check size of X and Y
        if len(X.shape) != 2:
            raise Exception('X is expected to be a 2-dimensional array of shape (n_samples, n_features)')            
        if len(Y.shape) != 2:
            raise Exception('Y is expected to be a 2-dimensional array of shape (n_samples, 1)')            
        if X.shape[0] != Y.shape[0]: 
            raise Exception('Different number of rows (number of samples) in X and Y') 

        # Add rules until no good rule can be found
        num_iter = 0
        best_obj = -1
        while (best_obj < 0) and (num_iter < self.max_rules):

            # Fit current best model
            model, r = self._fit_beta(X_m, R_m, Y, S)           
            
            # Perform column generation
            if self.optimizer == 'ilp': 
                best_rule, best_obj, best_rho = column_gen_ilp(X_m, R_m, r, relaxed=self.relaxed, 
                                gamma=self.gamma, lambda_0=self.lambda_0, lambda_1=self.lambda_1, 
                                optimality_tol=self.optimality_tol, feasibility_tol=self.feasibility_tol)
                
            elif self.optimizer == 'beam':
                # Don't pass intercept columns in X_m, R_m to beam search
                best_rule, best_obj, best_rho = column_gen_beam(X_m[:,1:], R_m[:, 1:], r,   
                                gamma=self.gamma, lambda_0=self.lambda_0, lambda_1=self.lambda_1, 
                                beam_width=self.beam_width, beam_depth=self.beam_depth)

                # Add intercept column back to the best rule
                best_rule = np.vstack([[0], best_rule])
            else:
                raise Exception('Unknown optimizer: %s' % self.optimizer)
                                                            
            if best_obj < 0:
                S = np.append(S, best_rule, axis=1)  
                if not self.silent:
                    print('Adding: ', best_rule.ravel(), 'Objective: %.2g' % best_obj, 'rho: %.2g' % best_rho.mean())

            num_iter += 1

        # Update stored rule set and compute assignments under new rules
        self.S = S
        a = rule_assignments(X_m, self.S)
        
        # Refit model under new rules
        # If a = empty array then fit dummy regressor, 
        #    and self.intercept_ = self.model.constant_

        if a.shape[1] < 2:
            if is_classifier(self):
                self.model = DummyClassifier(strategy="most_frequent").fit(a[:, 0:], Y)  # Or "constant" if a constant prediction is desired
            else:
                self.model = DummyRegressor(strategy="mean").fit(a[:, 0:], Y)  # Strategy depends on your use case
            
            self.coef_ = None
            
            # Handle intercept attribute based on model type
            if hasattr(self.model, "constant_"):  # DummyRegressor uses constant_
                self.intercept_ = self.model.constant_
            elif hasattr(self.model, "constant"):  # DummyClassifier with constant strategy uses constant
                self.intercept_ = self.model.constant
            else:
                self.intercept_ = None  # Set to None or some default value if constant is not applicable


            if not self.silent:
                print('WARNING: NO RULES WERE LEARNED!')
        else:
            # Refit current best model 
            model, _ = self._fit_beta(X_m, R_m, Y, self.S, self.reg_refit, reweight_coef=True)   
            self.model = model
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
            
        self.fitted = True

        if is_classifier(self):
            self.classes_ = unique_labels(Y)
            self.n_classes_ = len(self.classes_)

        return self

    def set_params(self, **params):
        if not params:
            return self

        if 'optimizer' in params:
            self.optimizer = params['optimizer']
        if 'max_rules' in params:
            self.max_rules = params['max_rules']
        if 'lambda_0' in params:
            self.lambda_0 = params['lambda_0']
        if 'lambda_1' in params:
            self.lambda_1 = params['lambda_1']
        if 'gamma' in params:
            self.gamma = params['gamma']
        if 'reg_refit' in params:
            self.reg_refit = params['reg_refit']
        if 'relaxed' in params:
            self.relaxed = params['relaxed']
        if 'optimality_tol' in params:
            self.optimality_tol = params['optimality_tol']
        if 'feasibility_tol' in params:
            self.feasibility_tol = params['feasibility_tol']
        if 'beam_width' in params:
            self.beam_width = params['beam_width']
        if 'beam_depth' in params:
            self.beam_depth = params['beam_depth']    
        if 'reg_rho' in params:
            self.reg_rho = params['reg_rho']
        if 'silent' in params:
            self.silent = params['silent']    

        return self

    def get_params(self, deep=True):
        return {"optimizer": self.optimizer,
                "max_rules": self.max_rules,
                "lambda_0": self.lambda_0,
                "lambda_1": self.lambda_1,
                "gamma": self.gamma,
                "reg_refit": self.reg_refit,
                "relaxed": self.relaxed,
                "optimality_tol": self.optimality_tol,
                "feasibility_tol": self.feasibility_tol,
                "beam_width": self.beam_width,
                "beam_depth": self.beam_depth,
                "reg_rho": self.reg_rho,
                "silent": self.silent,
                }


class MintyClassifier(ClassifierMixin, BaseMinty):
    "Minty for classification."

    def predict_proba(self, X):
        self.ensure_fitted()

        # Compute rule assignments
        X_m = np.c_[np.ones(X.shape[0]), X]
        X_m[np.isnan(X_m)] = 0

        a = rule_assignments(X_m, self.S)

        # self.model does not expect intercept column
        if a.shape[1] < 2:
            yp = self.model.predict_proba(a[:, 0:])  # DummyRegressor used
        else:
            yp = self.model.predict_proba(a[:, 1:])

        return yp


class MintyRegressor(RegressorMixin, BaseMinty):
    "Minty for regression."
