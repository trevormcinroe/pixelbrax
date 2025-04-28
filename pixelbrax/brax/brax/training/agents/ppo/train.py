# Copyright 2023 The Brax Authors.
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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

# train_fn = {
#   'inverted_pendulum': functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20,
#                                          reward_scaling=10, episode_length=1000,
#                                          normalize_observations=True, action_repeat=1,
#                                          unroll_length=5, num_minibatches=32,
#                                          num_updates_per_batch=4, discounting=0.97,
#                                          learning_rate=3e-4, entropy_cost=1e-2,
#                                          num_envs=2048, batch_size=1024, seed=1),
#   'inverted_double_pendulum': functools.partial(ppo.train, num_timesteps=20_000_000, num_evals=20,
#                                                 reward_scaling=10, episode_length=1000,
#                                                 normalize_observations=True, action_repeat=1,
#                                                 unroll_length=5, num_minibatches=32,
#                                                 num_updates_per_batch=4, discounting=0.97,
#                                                 learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048,
#                                                 batch_size=1024, seed=1),
#   'ant': functools.partial(ppo.train,  num_timesteps=50_000_000, num_evals=10,
#                            reward_scaling=10, episode_length=1000, normalize_observations=True,
#                            action_repeat=1, unroll_length=5, num_minibatches=32,
#                            num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
#                            entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
#   'humanoid': functools.partial(ppo.train,  num_timesteps=50_000_000, num_evals=10,
#                                 reward_scaling=0.1, episode_length=1000, normalize_observations=True,
#                                 action_repeat=1, unroll_length=10, num_minibatches=32,
#                                 num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4,
#                                 entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1),
#   'reacher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20,
#                                reward_scaling=5, episode_length=1000, normalize_observations=True,
#                                action_repeat=4, unroll_length=50, num_minibatches=32,
#                                num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,
#                                entropy_cost=1e-3, num_envs=2048, batch_size=256,
#                                max_devices_per_host=8, seed=1),
#   'humanoidstandup': functools.partial(ppo.train, num_timesteps=100_000_000, num_evals=20,
#                                        reward_scaling=0.1, episode_length=1000,
#                                        normalize_observations=True, action_repeat=1, unroll_length=15,
#                                        num_minibatches=32, num_updates_per_batch=8, discounting=0.97,
#                                        learning_rate=6e-4, entropy_cost=1e-2, num_envs=2048,
#                                        batch_size=1024, seed=1),
#   'hopper': functools.partial(sac.train, num_timesteps=6_553_600, num_evals=20,
#                               reward_scaling=30, episode_length=1000, normalize_observations=True,
#                               action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128,
#                               batch_size=512, grad_updates_per_step=64, max_devices_per_host=1,
#                               max_replay_size=1048576, min_replay_size=8192, seed=1),
#   'walker2d': functools.partial(sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5,
#                                 episode_length=1000, normalize_observations=True, action_repeat=1,
#                                 discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128,
#                                 grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576,
#                                 min_replay_size=8192, seed=1),
#   'halfcheetah': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20,
#                                    reward_scaling=1, episode_length=1000, normalize_observations=True,
#                                    action_repeat=1, unroll_length=20, num_minibatches=32,
#                                    num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,
#                                    entropy_cost=0.001, num_envs=2048, batch_size=512, seed=3),
#   'pusher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20,
#                               reward_scaling=5, episode_length=1000, normalize_observations=True,
#                               action_repeat=1, unroll_length=30, num_minibatches=16, num_updates_per_batch=8,
#                               discounting=0.95, learning_rate=3e-4,entropy_cost=1e-2, num_envs=2048,
#                               batch_size=512, seed=3),
# }[env_name]


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)
  return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    pixels: bool = False,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    logger=None
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    max_devices_per_host: maximum number of chips to use per host process
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy
      of the policy
    discounting: discounting rate
    seed: random seed
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    deterministic_eval: whether to run the eval with a deterministic policy
    network_factory: function that generates networks for policy and value
      functions
    progress_fn: a user-defined callback function for reporting/plotting metrics
    normalize_advantage: whether to normalize advantage estimate
    eval_env: an optional environment for eval only, defaults to `environment`
    policy_params_fn: a user-defined callback function that can be used for
      saving policy checkpoints
    randomization_fn: a user-defined callback function that generates randomized
      environments

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat)
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)
  print(f'Num training steps per epoch: {num_training_steps_per_epoch}')

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // local_device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )

  if isinstance(environment, envs.Env):
    print(f'In option A')
    wrap_for_training = envs.training.wrap
  else:
    print(f'In  option B')
    wrap_for_training = envs_v1.wrappers.wrap_for_training

  # The pixel bool modifies things here. Our impl of pixel env will already have vmapping and things like this
  # sorted.
  env = wrap_for_training(
      environment,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
      already_vmapped=pixels
  )

  reset_fn = jax.jit(jax.vmap(env.reset))
  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])
  env_state = reset_fn(key_envs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  # TODO: is this where we want to change the network to be CNN?
  ppo_network = network_factory(
      env_state.obs.shape[-1] if not pixels else env_state.pixels.shape,
      env.action_size,
      preprocess_observations_fn=normalize)

  make_policy = ppo_networks.make_inference_fn(ppo_network)

  optimizer = optax.adam(learning_rate=learning_rate)

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage)

  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

  def minibatch_step(
      carry, data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state)

    return (optimizer_state, params, key), metrics

  def sgd_step(carry, unused_t, data: types.Transition,
               normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    # print(f'data: {data.observation.shape} // {data.pixels.shape}')
    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    # print(f'data: {data.observation.shape} // {data.pixels.shape}')
    # qqq
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches)
    return (optimizer_state, params, key), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey],
      unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy(
        (training_state.normalizer_params, training_state.params.policy))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=('truncation',))
      return (next_state, next_key), data

    print(f'LEN OF UNROLL: {batch_size * num_minibatches // num_envs}')
    (state, _), data = jax.lax.scan(
        f, (state, key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    # print(f'DATA SHAPE: {data.observation.shape} // {data.pixels.shape}')
    # jax.debug.print('Completed an update...')
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                  data)
    # print(f'DATA SHAPE: {data.observation.shape} // {data.pixels.shape}')
    assert data.discount.shape[1:] == (unroll_length,)

    # qqq

    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        data.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params),
        (training_state.optimizer_state, training_state.params, key_sgd), (),
        length=num_updates_per_batch)

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step)
    return (new_training_state, state, new_key), metrics

  def training_epoch(training_state: TrainingState, state: envs.State,
                     key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step, (training_state, state, key), (),
        length=num_training_steps_per_epoch)
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State,
      key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_training_steps_per_epoch *
           env_step_per_training_step *
           max(num_resets_per_eval, 1)) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value))

  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          specs.Array(env_state.obs.shape[-1:], jnp.dtype('float32'))),
      env_steps=0)
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  if not eval_env:
    eval_env = environment
  if randomization_fn is not None:
    v_randomization_fn = functools.partial(
        randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
    )
  eval_env = wrap_for_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
      already_vmapped=pixels
  )

  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  print(f'============== ABOUT TO START {num_evals_after_init} STEP ==============')
  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.params.policy)),
        training_metrics={})
    logging.info(metrics)
    print(f'{training_state.env_steps}: {metrics["eval/episode_reward"]}')
    logger.log({'eval/returns': metrics["eval/episode_reward"], 'env_steps': training_state.env_steps})
    progress_fn(0, metrics)

  training_metrics = {}
  training_walltime = 0
  current_step = 0
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

      (training_state, env_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s),
          in_axes=(0, None))(key_envs, key_envs.shape[1])
      # TODO: move extra reset logic to the AutoResetWrapper.
      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.params.policy)),
          training_metrics)
      logging.info(metrics)
      print(f'{training_state.env_steps}: {metrics["eval/episode_reward"]}')
      logger.log({'eval/returns': metrics["eval/episode_reward"], 'env_steps': training_state.env_steps})
      progress_fn(current_step, metrics)
      params = _unpmap(
          (training_state.normalizer_params, training_state.params.policy))
      policy_params_fn(current_step, make_policy, params)

  total_steps = current_step
  assert total_steps >= num_timesteps

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.params.policy))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
