import functools

import numpy as np
from chainer.training import extension

_default_gamma = 0.99994


def _compute_next_value_triangular(t, value_range, step_size):
    v1, v2 = value_range

    cycle = np.floor(1 + t / (2 * step_size))
    x = np.abs(t / step_size - 2 * cycle + 1)
    value = v1 + (v2 - v1) * np.maximum(0, (1 - x))
    return value


def _compute_next_value_triangular2(t, value_range, step_size):
    v1, v2 = value_range

    cycle = np.floor(1 + t / (2 * step_size))
    x = np.abs(t / step_size - 2 * cycle + 1)
    value = v1 + (v2 - v1) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
    return value


def _compute_next_value_exp_range(t, value_range, step_size,
                                  gamma=_default_gamma):
    v1, v2 = value_range

    cycle = np.floor(1 + t / (2 * step_size))
    x = np.abs(t / step_size - 2 * cycle + 1)
    value = v1 + (v2 - v1) * np.maximum(0, (1 - x)) * (gamma ** t)
    return value


class CLR(extension.Extension):
    """Trainer extension to apply cyclical learning rate an optimizer attribute.

    This Cyclical learning Rate (CLR) is proposed in [#]_.

     .. [#] Smith, Leslie. Cyclical Learning Rates for Training Neural Networks.
        WACV 2017.

    Args:
        attr (str): Name of the attribute to apply.
        value_range (tuple of float): The first and last values of the
            attribute.
        step_size (int): The number of iterations per half cycle.
        policy (str): Policy to apply. The choices are 'triangular',
            'triangular2' and 'exp_range'.
        gamma (str): Base value for 'exp_range' policy. If other policy is used,
            This value is ignored.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.
    """

    _policy_choices = {
        'triangular': _compute_next_value_triangular,
        'triangular2': _compute_next_value_triangular2,
        'exp_range': _compute_next_value_exp_range
    }

    def __init__(self, attr, value_range, step_size, policy='triangular',
                 gamma=None, optimizer=None):
        self._attr = attr
        self._value_range = value_range
        self._step_size = step_size
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

        if policy not in self._policy_choices.keys():
            raise ValueError('not supported policy.')
        self._policy_func = self._policy_choices[policy]
        if policy == 'exp_range':
            gamma = gamma or _default_gamma
            self._policy_func = functools.partial(self._policy_func,
                                                  gamma=gamma)

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self._last_value is not None:
            value = self._last_value
        else:
            value = self._compute_next_value()
        self._update_value(optimizer, value)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._compute_next_value()
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _compute_next_value(self):
        return self._policy_func(self._t, self._value_range, self._step_size)

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
