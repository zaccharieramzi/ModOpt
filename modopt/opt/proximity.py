# -*- coding: utf-8 -*-

"""PROXIMITY OPERATORS

This module contains classes of proximity operators for optimisation

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import print_function
from builtins import range
import numpy as np

from modopt.base.types import check_callable
from modopt.signal.noise import thresh
from modopt.signal.svd import svd_thresh, svd_thresh_coef
from modopt.signal.positivity import positive
from modopt.math.matrix import nuclear_norm
from modopt.base.transform import cube2matrix, matrix2cube
from modopt.opt.algorithms import Condat
from modopt.opt.gradient import GradBasic
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold


class ProximityParent(object):

    def __init__(self, op, cost):

        self.op = op
        self.cost = cost

    @property
    def op(self):
        """Linear Operator

        This method defines the linear operator

        """

        return self._op

    @op.setter
    def op(self, operator):

        self._op = check_callable(operator)

    @property
    def cost(self):
        """Cost Contribution

        This method defines the proximity operator's contribution to the total
        cost

        """

        return self._cost

    @cost.setter
    def cost(self, method):

        self._cost = check_callable(method)


class IdentityProx(ProximityParent):
    """Identity Proxmity Operator

    This is a dummy class that can be used as a proximity operator

    Notes
    -----
    The identity proximity operator contributes 0.0 to the total cost

    """

    def __init__(self):

        self.op = lambda x: x
        self.cost = lambda x: 0.0


class ProjectionLInftyBall(ProximityParent):

    def __init__(self, thresh=1.0):

        self.op = self._op_method
        self.cost = self._cost_method
        self.thresh = thresh

    def _op_method(self, x, extra_factor=1.0):
        thresh = self.thresh * np.ones_like(x)
        proj_real =  np.maximum(-thresh, np.minimum(thresh, x.real))
        proj_im =  np.maximum(-thresh, np.minimum(thresh, x.imag))
        return proj_real + 1j * proj_im

    def _cost_method(self, x):
        if np.max(x.real) > self.thresh or np.max(x.imag) > self.thresh:
            return np.Inf
        else:
            return 0.0


class Positivity(ProximityParent):
    """Positivity Proximity Operator

    This class defines the positivity proximity operator

    """

    def __init__(self):

        self.op = lambda x: positive(x)
        self.cost = self._cost_method

    def _cost_method(self, *args, **kwargs):
        """Calculate positivity component of the cost

        This method returns 0 as the posivituty does not contribute to the
        cost.

        Returns
        -------
        float zero

        """

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - Min (X):', np.min(args[0]))

        return 0.0


class SparseThreshold(ProximityParent):
    """Threshold proximity operator

    This class defines the threshold proximity operator

    Parameters
    ----------
    linear : class
        Linear operator class
    weights : np.ndarray
        Input array of weights
    thresh_type : str {'hard', 'soft'}, optional
        Threshold type (default is 'soft')

    """

    def __init__(self, linear, weights, thresh_type='soft'):

        self._linear = linear
        self.weights = weights
        self._thresh_type = thresh_type
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, data, extra_factor=1.0):
        """Operator Method

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray thresholded data

        """

        threshold = self.weights * extra_factor

        return thresh(data, threshold, self._thresh_type)

    def _cost_method(self, *args, **kwargs):
        """Calculate sparsity component of the cost

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Returns
        -------
        float sparsity cost component

        """

        cost_val = np.sum(np.abs(self.weights * self._linear.op(args[0])))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - L1 NORM (X):', cost_val)

        return cost_val


class LowRankMatrix(ProximityParent):
    r"""Low-rank proximity operator

    This class defines the low-rank proximity operator

    Parameters
    ----------
    thresh : float
        Threshold value
    treshold_type : str {'hard', 'soft'}
        Threshold type (options are 'hard' or 'soft')
    lowr_type : str {'standard', 'ngole'}
        Low-rank implementation (options are 'standard' or 'ngole')
    operator : class
        Operator class ('ngole' only)

    Examples
    --------
    >>> from modopt.opt.proximity import LowRankMatrix
    >>> a = np.arange(9).reshape(3, 3).astype(float)
    >>> inst = LowRankMatrix(10.0, thresh_type='hard')
    >>> inst.op(a)
    array([[[  2.73843189,   3.14594066,   3.55344943],
            [  3.9609582 ,   4.36846698,   4.77597575],
            [  5.18348452,   5.59099329,   5.99850206]],

           [[  8.07085295,   9.2718846 ,  10.47291625],
            [ 11.67394789,  12.87497954,  14.07601119],
            [ 15.27704284,  16.47807449,  17.67910614]]])
    >>> inst.cost(a, verbose=True)
     - NUCLEAR NORM (X): 469.391329425
    469.39132942464983

    """

    def __init__(self, thresh, thresh_type='soft',
                 lowr_type='standard', operator=None):

        self.thresh = thresh
        self.thresh_type = thresh_type
        self.lowr_type = lowr_type
        self.operator = operator
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, data, extra_factor=1.0):
        """Operator

        This method returns the input data after the singular values have been
        thresholded

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray SVD thresholded data

        """

        # Update threshold with extra factor.
        threshold = self.thresh * extra_factor

        if self.lowr_type == 'standard':
            data_matrix = svd_thresh(cube2matrix(data), threshold,
                                     thresh_type=self.thresh_type)

        elif self.lowr_type == 'ngole':
            data_matrix = svd_thresh_coef(cube2matrix(data), self.operator,
                                          threshold,
                                          thresh_type=self.thresh_type)

        new_data = matrix2cube(data_matrix, data.shape[1:])

        # Return updated data.
        return new_data

    def _cost_method(self, *args, **kwargs):
        """Calculate low-rank component of the cost

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Returns
        -------
        float low-rank cost component

        """

        cost_val = self.thresh * nuclear_norm(cube2matrix(args[0]))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - NUCLEAR NORM (X):', cost_val)

        return cost_val


class LinearCompositionProx(ProximityParent):
    """Proximity operator of a linear composition

    This class defines the proximity operator of a function given by
    a composition between an initial function whose proximity operator is known
    and an orthogonal linear function.

    Parameters
    ----------
    linear_op : class instance
        Linear operator class
    prox_op : class instance
        Proximity operator class
    """
    def __init__(self, linear_op, prox_op):
        self.linear_op = linear_op
        self.prox_op = prox_op
        self.op = self._op_method
        self.cost = self._cost_method

    def _op_method(self, data, extra_factor=1.0):
        r"""Operator method

        This method returns the scaled version of the proximity operator as
        given by Lemma 2.8 of [CW2005].

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray result of the scaled proximity operator
        """
        return self.linear_op.adj_op(
            self.prox_op.op(self.linear_op.op(data), extra_factor=extra_factor)
        )

    def _cost_method(self, *args, **kwargs):
        """Calculate the cost function associated to the composed function

        Returns
        -------
        float the cost of the associated composed function
        """
        return self.prox_op.cost(self.linear_op.op(args[0]), **kwargs)


class LinearCompositionIterativeProx(LinearCompositionProx):
    """ Proximity operator of the composition of a function with a linear
    function computed in an iterative fashion
    """
    def __init__(
            self,
            linear_op,
            prox_op,
            max_precision_level=None,
            solver_sigma=10.0,
        ):
        super(LinearCompositionIterativeProx, self).__init__(linear_op, prox_op)
        self.max_precision_level = max_precision_level
        self.solver_sigma = solver_sigma

    def _op_method(self, data, extra_factor=1.0, precision_level=None):
        """Solve the proximity oeprator primal optimisation problem
        """
        # setting the number of iterations for the solver
        max_iter = 100
        if precision_level is not None:
            max_iter = precision_level
        if self.max_precision_level is not None:
            max_iter = min(max_iter, self.max_precision_level)

        # defining the solver operators
        dual_init = self.linear_op.op(data)
        gradient_op = GradBasic(data, Identity().op, Identity().adj_op)
        prox_op_primal = Identity()
        prox_op_dual = SparseThreshold(self.linear_op, None, thresh_type="soft")
        prox_op_dual.weights = extra_factor * np.ones_like(dual_init)

        # solver params
        lipschitz_cst = 1.0  # because identity
        sigma = self.solver_sigma
        eps = 1.0e-8  # avoid numerical errors
        linear_op_norm = self.linear_op.l2norm(data.shape)
        tau = 1.0 / (lipschitz_cst/2 + sigma * linear_op_norm **2 + eps)
        # NOTE: maybe allow rho to be set
        rho = 1.0

        # solver definition
        opt = Condat(
            x=data,
            y=dual_init,
            grad=gradient_op,
            prox=prox_op_primal,
            prox_dual=prox_op_dual,
            linear=self.linear_op,
            cost=None,
            sigma=sigma,
            tau=tau,
            rho=rho,
            auto_iterate=False,
            metric_call_period=0,
            metrics=None,
        )
        # run the computation
        opt.iterate(max_iter=max_iter)
        x_final = opt.x_final

        return x_final



class ProximityCombo(ProximityParent):
    r"""Proximity Combo

    This class defines a combined proximity operator

    Parameters
    ----------
    operators : list
        List of proximity operator class instances

    Examples
    --------
    >>> from modopt.opt.proximity import ProximityCombo, ProximityParent
    >>> a = ProximityParent(lambda x: x ** 2, lambda x: x ** 3)
    >>> b = ProximityParent(lambda x: x ** 4, lambda x: x ** 5)
    >>> c = ProximityCombo([a, b])
    >>> c.op([2, 2])
    array([4, 16], dtype=object)
    >>> c.cost([2, 2])
    40

    """

    def __init__(self, operators):

        operators = self._check_operators(operators)
        self.operators = operators
        self.op = self._op_method
        self.cost = self._cost_method

    def _check_operators(self, operators):
        """ Check Inputs

        This method cheks that the input operators and weights are correctly
        formatted

        Parameters
        ----------
        operators : list, tuple or np.ndarray
            List of linear operator class instances

        Returns
        -------
        np.array operators

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not isinstance(operators, (list, tuple, np.ndarray)):
            raise TypeError('Invalid input type, operators must be a list, '
                            'tuple or numpy array.')

        operators = np.array(operators)

        if not operators.size:
            raise ValueError('Operator list is empty.')

        for operator in operators:
            if not hasattr(operator, 'op'):
                raise ValueError('Operators must contain "op" method.')
            if not hasattr(operator, 'cost'):
                raise ValueError('Operators must contain "cost" method.')
            operator.op = check_callable(operator.op)
            operator.cost = check_callable(operator.cost)

        return operators

    def _op_method(self, data, extra_factor=1.0):
        """Operator

        This method returns the result of applying all of the proximity
        operators to the data

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray result

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data[i], extra_factor=extra_factor)

        return res

    def _cost_method(self, *args, **kwargs):
        """Calculate combined proximity operator components of the cost

        This method returns the sum of the cost components from each of the
        proximity operators

        Returns
        -------
        float combinded cost components

        """

        return np.sum([operator.cost(data) for operator, data in
                       zip(self.operators, args[0])])
