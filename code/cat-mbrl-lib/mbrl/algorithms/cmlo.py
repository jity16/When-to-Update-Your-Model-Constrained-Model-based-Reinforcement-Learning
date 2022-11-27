# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
import ipdb
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import csv
import math

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

import datetime

def init(cwd):
    global opt
    
    opt = {}
    opt['time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    opt['description'] = os.environ['des']
    opt['algorithm'] = os.environ['algo']
    opt['freq_type'] = os.environ['freq']
    opt['filename'] = opt['description'] + "_" + opt['algorithm'] + "_" + opt["freq_type"] + "_" + opt['time']
    opt['log_dir'] = os.path.join(cwd, 'log_dir', opt['filename'] + "_" + os.getcwd()[-6:])
    
    from torch.utils.tensorboard import SummaryWriter
    global writer
    writer = SummaryWriter(opt['log_dir'])
    
    os.system("ln -s %s %s" % ("../../exp/mbpo" + os.getcwd().split("exp/mbpo")[-1], os.path.join(opt['log_dir'], 'exp')))


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):

    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    global opt
    global writer
    trajectory_reward = 0
    explore_steps = 0


    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    work_dir = work_dir or os.getcwd()
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None

    compute_interval = 50
    alpha = 2.0
    lower_bound = 150
    upper_bound = 500
    writer.add_text('Hyperparamters', "compute_interval: %d, alpha: %f, lower_bound: %d, upper_bound: %d"%(compute_interval, alpha, lower_bound, upper_bound), 1)
    convexhull = []
    target_dims = 3
    convexhull_batchsize = 1000
    triggered_flag = False
    event_value = []
    sum_of_event = 0

    event_csvfile_path = os.path.join(opt["log_dir"] , "c%d_alpha_%f_event.csv"%(compute_interval, alpha))
    csvfile = open(event_csvfile_path, 'w', newline='')
    fieldnames = ["env_steps", "event_value", "interval"]
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    last_train_step = 0
    
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer

        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs, done = None, False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            action, next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            trajectory_reward += reward
            if done:
                writer.add_scalar('Reward/Train_Reward', trajectory_reward, env_steps)
                trajectory_reward = 0

            if opt['freq_type'] == "fixed":
                event_intervals = cfg.overrides.freq_train_model
            else:
                if opt['freq_type'] == "incre":
                    freq_config = cfg.overrides.freq_incre
                elif opt['freq_type'] == "decre":
                    freq_config = cfg.overrides.freq_decre
                elif opt['freq_type'][:8] == "custom__":
                    freq_name = opt['freq_type'][8:]
                    assert freq_name != ""
                    freq_config = cfg.overrides.get("freq__" + freq_name, None)
                    assert freq_config != None
                
                event_intervals = freq_config.event_intervals[-1]
                for i in range(len(freq_config.event_steps)):
                    if env_steps <= freq_config.event_steps[i]:
                        event_intervals = freq_config.event_intervals[i]
                        break

            if (env_steps + 1) > event_intervals:
                model_len = len(model_env.dynamics_model.model.elite_models) if model_env.dynamics_model.model.elite_models is not None else len(model_env.dynamics_model.model)
                model_action = np.repeat(np.expand_dims(action,axis=0),model_len,axis=0)
                model_obs = np.repeat(np.expand_dims(obs,axis=0),model_len,axis=0)
                model_state = {'obs':model_obs, 'propagation_indices':None}
                pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(model_action, model_state, sample=True) #? set stochastic model predictions
                real_next_obs = np.repeat(np.expand_dims(next_obs,axis=0),model_len,axis=0)
                residual_obs = pred_next_obs - real_next_obs
                residual_L2_norm = np.linalg.norm(residual_obs,axis=1)
                step_L2_average = np.average(residual_L2_norm)
                step_L2_range = residual_L2_norm.max()-residual_L2_norm.min()
                step_L2_std = np.std(residual_L2_norm)
                
                sum_L2_average = sum_L2_average + step_L2_average     
                explore_steps += 1
                slide_L2_average = sum_L2_average/explore_steps

                writer.add_scalar('L2_norm/sum_of_interval_average', sum_L2_average, env_steps)
                writer.add_scalar('L2_norm/slide_average', slide_L2_average, env_steps)
                writer.add_scalar('L2_norm/step_average', step_L2_average, env_steps)
                writer.add_scalar('L2_norm/step_range', step_L2_range, env_steps)
                writer.add_scalar('L2_norm/step_std', step_L2_std, env_steps)

            if (env_steps + 1) % compute_interval == 0:
                convexhull_batch = replay_buffer.sample(convexhull_batchsize)
                batch_obs, *_ = cast(mbrl.types.TransitionBatch, convexhull_batch).astuple()
                pca = PCA(n_components=target_dims)
                X_pca = pca.fit_transform(batch_obs)
                convexhull.append(ConvexHull(X_pca).volume)
                writer.add_scalar('ConvexHull/volume', convexhull[-1], env_steps)
                if len(convexhull) > 1 and (env_steps > event_intervals):
                    delta_convexhull = convexhull[-1]/convexhull[-2]
                    event_value.append(delta_convexhull*slide_L2_average)
                    sum_of_event += math.log(delta_convexhull*slide_L2_average + 1.0)
                    writer.add_scalar('ConvexHull/relative_volume', delta_convexhull, env_steps)
                    writer.add_scalar('ConvexHull/relative_volume*slide_L2', event_value[-1], env_steps)
                    writer.add_scalar('Event/sum_of_log_sum', sum_of_event, env_steps)
                    if (len(event_value) > 1) and (explore_steps >= lower_bound):
                        if sum_of_event > alpha:
                            triggered_flag = True
                            sum_of_event  = 0 
                            csvwriter.writerow({"env_steps":env_steps, "event_value":event_value[-1], "interval": explore_steps})
                            writer.add_scalar('Event/interval', explore_steps, env_steps)
                            explore_steps = 0 


                
            # --------------- Model Training -----------------
            if (env_steps + 1) == event_intervals:
                triggered_flag = True
                sum_of_event  = 0 
                csvwriter.writerow({"env_steps":env_steps, "event_value":0.0, "interval": explore_steps})
                writer.add_scalar('Event/interval', explore_steps, env_steps)
                explore_steps = 0
            if explore_steps >= upper_bound:
                triggered_flag = True
                sum_of_event  = 0 
                if len(event_value) > 1:
                    csvwriter.writerow({"env_steps":env_steps, "event_value":event_value[-1], "interval": explore_steps})
                    writer.add_scalar('Event/interval', explore_steps, env_steps)
                else:
                    csvwriter.writerow({"env_steps":env_steps, "event_value":0.0, "interval": explore_steps})
                    writer.add_scalar('Event/interval', explore_steps, env_steps)
                explore_steps = 0 
            
            if triggered_flag == True:
                csvfile.flush()
                triggered_flag = False
                sum_of_event = 0
                sum_L2_average = step_L2_average = step_L2_range = step_L2_std = 0
                explore_steps = 0
                if env_steps +1 > event_intervals:
                    writer.add_scalar('L2_norm/interval_average', slide_L2_average, env_steps)
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
                
                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            rollout_batch_size = (
                cfg.overrides.effective_model_rollouts_per_step
            )
            trains_per_epoch = int(
                cfg.overrides.epoch_length
            )

            new_sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
            new_sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer

            if new_sac_buffer_capacity != sac_buffer_capacity:
                sac_buffer_capacity = new_sac_buffer_capacity
                sac_buffer = maybe_replace_sac_buffer(
                    sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
                )
                print("step %d, sac_buffer_capacity %d" % (env_steps, sac_buffer_capacity))

            # --------- Rollout new model and store imagined trajectories --------
            # Batch all rollouts for the next freq_train_model steps together

            if (env_steps + 1) >= event_intervals:
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                writer.add_scalar('Reward/Test_Reward', avg_reward, env_steps)
                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    video_recorder.save(f"{epoch}.gif")
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
