
import argparse
import torch
import time
import os
import numpy as np
from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = True  # torch.cuda.is_available()

# Force CUDA initialization early
if torch.cuda.is_available():
    torch.cuda.init()
    print(f" CUDA initialized: {torch.cuda.get_device_name(0)}")
else:
    print(" No CUDA available, using CPU")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            np.random.seed(seed + rank * 1000)
            env.reset(seed=seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    print("\n[DEBUG] ========== ENVIRONMENT INFO ==========")
    # Access the underlying environment
    #base_env = env.envs[0]  # Get the first environment from the wrapper
    #print(f"[DEBUG] Number of agents: {base_env.n}")
    #print(f"[DEBUG] Observation spaces: {base_env.observation_space}")
    #print(f"[DEBUG] Action spaces: {base_env.action_space}")
    # base_env used
    #for i, (obs_space, act_space) in enumerate(zip(base_env.observation_space, base_env.action_space)):
     #   print(f"[DEBUG] Agent {i}: obs_shape={obs_space.shape}, action_shape={act_space.shape}")
      #  for i, (obs_space, act_space) in enumerate(zip(env.observation_space, env.action_space)):
       #     print(f"[DEBUG] Agent {i}: obs_shape={obs_space.shape}, action_shape={act_space.shape}")
    print("\n[DEBUG] ========== INITIALIZING MADDPG ==========")
    maddpg = MADDPG.init_from_env_with_delay(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  delay_step = 3)
    print(f"[DEBUG] MADDPG initialized with {maddpg.nagents} agents")
    for i, agent in enumerate(maddpg.agents):
        print(f"[DEBUG] Agent {i} policy input dim: {agent.policy.in_fn}")
        print(f"[DEBUG] Agent {i} policy first layer input features: {agent.policy.in_fn.num_features}")
        print(f"[DEBUG] Agent {i} policy BatchNorm running_mean size: {agent.policy.in_fn.running_mean.shape}")
    delay_step = 3
    #base_env used
    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [env.observation_space[i].shape[0] + env.action_space[i].shape[0] * delay_step 
         for i in range(maddpg.nagents)],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
         for acsp in env.action_space]
    )
    #print(f"\n[DEBUG] Replay buffer obs dims: {[base_env.observation_space[i].shape[0] + base_env.action_space[i].shape[0] * delay_step for i in range(maddpg.nagents)]}")
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        #base_env = env.envs[0]
        #print("[DEBUG] Agents in environment:", base_env.agents)
        #print("[DEBUG] Observation spaces per agent:")
        #for agent in base_env.agents:
         #   obs_space = base_env.observation_space(agent)
          #  print(f"  Agent {agent}: Observation space: {obs_space}, shape: {getattr(obs_space, 'shape', None)}, type: {type(obs_space)}")

        obs = env.reset()

        # FIX: Numpy is converting to object dtype when creating the array
        # We need to reconstruct it properly while maintaining array structure
        if len(obs) > 0 and len(obs[0]) > 0:
            if obs[0][0].dtype == object:
                print("[FIX] Detected object dtype, converting to float32...")
                obs_fixed = np.empty(len(obs), dtype=object)
                for env_idx in range(len(obs)):
                    # Keep as numpy array, not list
                    obs_fixed[env_idx] = np.array([np.asarray(obs[env_idx][a_i], dtype=np.float32) 
                                                   for a_i in range(len(obs[env_idx]))], dtype=object)
                obs = obs_fixed
                print("[FIX] Conversion complete")
        
        print(f"[DEBUG] After reset, obs type: {type(obs)}, len: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
        print(f"[DEBUG] obs[0] type: {type(obs[0])}, len: {len(obs[0])}")
        print(f"[DEBUG] obs shape: {obs.shape}")  # Add this to check structure
        for i, o in enumerate(obs[0]):
            print(f"[DEBUG] obs[0][{i}] shape: {o.shape}, dtype: {o.dtype}")
        if USE_CUDA:
            maddpg.prep_rollouts(device='gpu')
        else:
            maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # base_env used
        last_agent_actions = []
        for env_idx in range(config.n_rollout_threads):
            zero_agent_actions = [np.zeros(env.action_space[i].shape[0]) for i in range(maddpg.nagents)]
            print(f"\n[DEBUG] zero_agent_actions: {[a.shape for a in zero_agent_actions]}")
        
            last_agent_actions.append([zero_agent_actions.copy() for _ in range(delay_step)])
            print(f"[DEBUG] last_agent_actions length: {len(last_agent_actions)}")
        # Append delayed actions to observations for ALL environments
        for env_idx in range(config.n_rollout_threads):
            for a_i in range(len(obs[env_idx])):
                agent_obs = obs[env_idx][a_i]
                for delay_idx in range(delay_step):
                    agent_obs = np.append(agent_obs, last_agent_actions[env_idx][delay_idx][a_i])
                obs[env_idx][a_i] = agent_obs
        
        print(f"[DEBUG] After appending delays:")
        for env_idx in range(min(2, config.n_rollout_threads)):  # Print first 2 envs
            print(f"[DEBUG]   Env {env_idx}: agent shapes = {[obs[env_idx][a].shape for a in range(len(obs[env_idx]))]}")
        
        print("\n[DEBUG] Final obs shapes after appending:")
        for i, o in enumerate(obs[0]):
            print(f"[DEBUG] obs[0][{i}] final shape: {o.shape}")
        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            print(f"[DEBUG] Calling maddpg.step...")
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            print(f"[DEBUG] maddpg.step completed")
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = []
            for env_idx in range(config.n_rollout_threads):
                if delay_step == 0:
                    env_actions = [agent_actions[a_i][env_idx] for a_i in range(maddpg.nagents)]
                else:
                    # Get current actions for this environment
                    agent_actions_tmp = [agent_actions[a_i][env_idx] for a_i in range(maddpg.nagents)]
                    # Use oldest delayed action
                    env_actions = last_agent_actions[env_idx][0]
                    # Update delay buffer
                    last_agent_actions[env_idx] = last_agent_actions[env_idx][1:]
                    last_agent_actions[env_idx].append(agent_actions_tmp)
                
                actions.append(env_actions)
            next_obs, rewards, dones, infos = env.step(actions)
            for env_idx in range(config.n_rollout_threads):
                for a_i in range(len(next_obs[env_idx])):
                    agent_obs = next_obs[env_idx][a_i]
                    for delay_idx in range(delay_step):
                        # Apply scaling if needed (keeping your original logic)
                        if a_i == 2:
                            agent_obs = np.append(agent_obs, 4*last_agent_actions[env_idx][delay_idx][a_i])
                        else:
                            agent_obs = np.append(agent_obs, 3*last_agent_actions[env_idx][delay_idx][a_i])
                    next_obs[env_idx][a_i] = agent_obs
            # Scale agent actions for replay buffer 
            scaled_agent_actions = []
            for a_i in range(maddpg.nagents):
                if a_i == 2:
                    scaled_agent_actions.append(agent_actions[a_i] * 4)
                else:
                    scaled_agent_actions.append(agent_actions[a_i] * 3)
            #agent_actions[0] = agent_actions[0]*3
            #agent_actions[1] = agent_actions[1]*3
            #agent_actions.append(agent_actions[1]*4)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
    

            
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents - 1): #do not update the runner
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_adversaries()
                if USE_CUDA:
                    maddpg.prep_rollouts(device='gpu')
                else:
                    maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            print(f"Episode {ep_i+1}, Agent {a_i} total reward: {a_ep_rew}")
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
#     parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
