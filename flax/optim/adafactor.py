# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adafactor Optimizer.

This is a so-called "1+epsilon" optimizer that is extremely memory efficient
compared to Adam, and has had wide success when applied to large-scale training
of attention-based models.
"""
from .. import struct
from .base import OptimizerDef

import jax.numpy as jnp

import numpy as onp


@struct.dataclass
class _AdafactorHyperParams:
  learning_rate: onp.ndarray
  factored: bool
  multiply_by_parameter_scale: bool
  do_clipping: bool
  do_momentum: bool
  beta1: onp.ndarray
  decay_rate: onp.ndarray
  clipping_threshold: onp.ndarray
  weight_decay_rate: onp.ndarray
  epsilon1: onp.ndarray
  epsilon2: onp.ndarray


@struct.dataclass
class _AdafactorParamState:
  v_row: onp.ndarray  # used in normal factored version
  v_col: onp.ndarray  #
  v: onp.ndarray  # only used without factoring
  m: onp.ndarray  # only used with momentum


# TODO(levskaya): this matches the mesh-tf implementation, but consider
#                 removing the unfactored-version and/or momentum.
class Adafactor(OptimizerDef):
  """Adafactor optimizer."""

  def __init__(self,
               learning_rate=None,  # ~ 0.05
               factored=True,
               multiply_by_parameter_scale=True,
               do_clipping=True,
               do_momentum=False,
               beta1=0.0,
               decay_rate=0.8,
               clipping_threshold=1.0,
               weight_decay_rate=1e-5,
               epsilon1=1e-30,
               epsilon2=1e-3):
    """Constructor for the Adafactor optimizer.

    Adafactor is described in https://arxiv.org/abs/1804.04235.

    Args:
      learning_rate: float: learning rate.  NB: the natural scale for adafactor
        LR is markedly different from Adam, one doesn't use the 1/sqrt(hidden)
        correction for this optimizer with attention-based models.
      factored: boolean: whether to use factored second-moment estimator for 2d
        variables.
      multiply_by_parameter_scale: boolean: if True, then scale provided
        learning_rate by parameter norm. if False, provided learning_rate is
        absolute step size.
      do_clipping: whether to clip gradients; if True, set clipping_threshold.
      do_momentum: whether to use momentum; if True, set beta1.
      beta1: a float value between 0 and 1, enables momentum and uses extra
        memory if nonzero!  Off by default.
      decay_rate: float: controls second-moment exponential decay schedule.
      clipping_threshold: an optional float >= 1, if None no update clipping.
      weight_decay_rate: rate at which to decay weights.
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
    """
    hyper_params = _AdafactorHyperParams(
        learning_rate, factored, multiply_by_parameter_scale,
        do_clipping, do_momentum, beta1, decay_rate, clipping_threshold,
        weight_decay_rate, epsilon1, epsilon2)
    super().__init__(hyper_params)

  @staticmethod
  def _decay_rate_pow(i, exponent=0.8):
    """Default Adafactor second-moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return 1.0 - t**(-exponent)

  def init_param_state(self, param):
    shape = param.shape
    state = {k: jnp.zeros(()) for k in ['v_row', 'v_col', 'v', 'm']}
    if self.hyper_params.factored and len(shape) >= 2:
      state['v_row'] = jnp.zeros(shape[:-1], dtype=jnp.float32)
      state['v_col'] = jnp.zeros(shape[:-2] + shape[-1:], dtype=jnp.float32)
    else:
      state['v'] = jnp.zeros_like(param)
    if self.hyper_params.do_momentum:
      state['m'] = jnp.zeros_like(param)
    return _AdafactorParamState(**state)

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    learning_rate = hyper_params.learning_rate
    beta1 = hyper_params.beta1
    decay_rate = hyper_params.decay_rate
    clipping_threshold = hyper_params.clipping_threshold
    weight_decay_rate = hyper_params.weight_decay_rate
    epsilon1 = hyper_params.epsilon1
    epsilon2 = hyper_params.epsilon2

    updates = {k: jnp.zeros(()) for k in ['v_row', 'v_col', 'v', 'm']}
    decay_rate = self._decay_rate_pow(step, exponent=decay_rate)
    update_scale = learning_rate
    if self.hyper_params.multiply_by_parameter_scale:
      update_scale *= jnp.maximum(
          jnp.sqrt(jnp.mean(param * param)), epsilon2)
    mixing_rate = 1.0 - decay_rate

    grad_sqr = grad * grad + epsilon1
    if self.hyper_params.factored and len(param.shape) >= 2:
      new_v_row = (
          decay_rate * state.v_row + mixing_rate * jnp.mean(grad_sqr, axis=-1))
      new_v_col = (
          decay_rate * state.v_col + mixing_rate * jnp.mean(grad_sqr, axis=-2))
      updates['v_row'] = new_v_row
      updates['v_col'] = new_v_col
      row_col_mean = jnp.mean(new_v_row, axis=-1, keepdims=True)
      row_factor = (new_v_row / row_col_mean) ** -0.5
      col_factor = (new_v_col) ** -0.5
      y = (
          grad * jnp.expand_dims(row_factor, axis=-1) *
          jnp.expand_dims(col_factor, axis=-2))
    else:
      new_v = decay_rate * state.v + mixing_rate * grad_sqr
      updates['v'] = new_v
      y = grad * (new_v)**-0.5

    if self.hyper_params.do_clipping:
      clipping_denom = (
          jnp.maximum(1.0, jnp.sqrt(jnp.mean(y * y)) / clipping_threshold))
      y /= clipping_denom

    subtrahend = update_scale * y
    if self.hyper_params.do_momentum:
      new_m = beta1 * state.m + (1.0 - beta1) * subtrahend
      subtrahend = new_m
      updates['m'] = new_m

    new_param = (1 - weight_decay_rate) * param - subtrahend
    new_state = _AdafactorParamState(**updates)
    return new_param.astype(param.dtype), new_state
