from typing import Union, Callable
import numpy as np
from copy import copy

"""Penalty functions."""


def power(x: Union[float, np.array], x_0: float, power: float, weight: float) -> Union[float, np.array]:
    """Calculates the power function as penalty function.

    Parameters
    ----------
    x : Union[float, np.array]
    x_0 : floats
        The value of x for which the penalty is zero
    power : float
        Power of the penalty function
    weight : float
        Multipliying coefficient for the penalty function

    Returns
    -------
    Union[float, np.array]
    """

    x_ = copy(x)
    x_0_ = copy(x_0)
    x_ *= -1.0
    x_0_ *= -1.0

    if not np.isscalar(x):
        x_vec = copy(x_) - x_0_
        y = np.zeros(len(x_vec))

        for i in range(len(x_vec)):
            if x_vec[i] <= 0:
                y[i] = 0
            else:
                y[i] = x_vec[i] ** power
    else:
        x_ -= x_0_

        y = 0.0
        if x_ <= 0:
            y = 0
        else:
            y = x_ ** power

    return y * weight


def smooth_square_root(
    x: Union[float, np.array], x_0: float, epsilon: float = 0.1, weight: float = 1,
) -> Union[float, np.array]:
    """Calculates the smooth approximation to the square-root exact penalty function.
    
    Parameters
    ----------
    x : Union[float, np.array]
    x_0 : floats
        The value of x for which the penalty is zero
    epsilon : float = 0.1
        Smoothing parameter (the bigger, the smoother)
    weight : float = 1
        Multipliying coefficient for the penalty function

    Returns
    -------
    np.array or float

    References
    ----------
    [1] Y. Duan and S. Lian, “Smoothing Approximation to the Square-Root 
        Exact Penalty Function,” Journal of Systems Science and Information, 
        vol. 4, no. 1, pp. 87–96, Feb. 2016, doi: 10.1515/JSSI-2016-0087.
    """
    x_ = copy(x)
    x_0_ = copy(x_0)
    x_ *= -1.0
    x_0_ *= -1.0

    if not np.isscalar(x):
        x_vec = copy(x_) - x_0_
        y = np.zeros(len(x_vec))

        for i in range(len(x_vec)):
            if x_vec[i] <= 0:
                y[i] = 2.0 / 3.0 * epsilon**0.5
            elif x_vec[i] <= epsilon and x_vec[i] > 0.0:
                y[i] = (
                    1.0 / 3.0 * epsilon**-1.0 * x_vec[i] ** 1.5
                    + 2.0 / 3.0 * epsilon**0.5
                )
            else:
                y[i] = x_vec[i] ** 0.5
    else:
        x_ -= x_0_

        y = 0.0
        if x_ <= 0:
            y = 2.0 / 3.0 * epsilon**0.5
        elif x_ <= epsilon and x_ > 0.0:
            y = 1.0 / 3.0 * epsilon**-1.0 * x_**1.5 + 2.0 / 3.0 * epsilon**0.5
        else:
            y = x_**0.5
    y_0 = 2.0 / 3.0 * epsilon**0.5

    return (y - y_0) * weight


def penalty(
    x: Union[np.array, float],
    lb: float,
    ub: float,
    fct: Callable = smooth_square_root,
    **args_fct: dict
) -> Union[np.array, float]:
    """Calculates the penalty value for constraint.
    Parameters
    ----------
    x : Union[np.array, float]
        value(s) to calculate the penalty for
    lb : float
        lower bound
    ub : float
        upper bound
    fct : Callable, optional
        penalty function, by default smooth_square_root
    args_fct : dict
        arguments for the penalty function
    Returns
    -------
    Union[np.array, float]
        penalty value(s)
    """
    return fct(x, lb, **args_fct) + fct(-x, -ub, **args_fct)