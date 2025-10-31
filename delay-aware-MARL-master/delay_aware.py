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

def compute_virtual_action(action_buffer, delay_float):
    """
    Compute virtual effective action for non-integral delays.
    
    Args:
        action_buffer: List of past actions [newest, ..., oldest]
        delay_float: Non-integral delay (e.g., 2.3)
    
    Returns:
        Virtual effective action: (1-f)*a[I+1] + f*a[I]
    """
    I = int(np.floor(delay_float))  # Integer part
    f = delay_float - I  # Fractional part
    
    if f == 0:
        # Pure integral delay
        return action_buffer[I]
    else:
        # Non-integral delay: interpolate between two actions
        # action_buffer[I] is the action I steps ago
        # action_buffer[I+1] is the action I+1 steps ago
        return (1 - f) * action_buffer[I] + f * action_buffer[I + 1]

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
    print(f"[DEBUG] Delay step: {config.delay_step} (I={int(np.floor(config.delay_step))}, f={config.delay_step - int(np.floor(config.delay_step))})")
    
    print("\n[DEBUG] ========== INITIALIZING MADDPG ==========")
    
    # Calculate buffer size needed (ceiling of delay)
    delay_buffer_size = int(np.ceil(config.delay_step)) + 1
    
    maddpg = MADDPG.init_from_env_with_delay(
        env, 
        agent_alg=config.agent_alg,
        adversary_alg=config.adversary_alg,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        delay_step=config.delay_step,
        use_sigmoid=True
    )
    
    print(f"[DEBUG] MADDPG initialized with {maddpg.nagents} agents")
    print(f"[DEBUG] Delay buffer size: {delay_buffer_size}")
    for i, agent in enumerate(maddpg.agents):
        print(f"[DEBUG] Agent {i} policy input dim: {agent.policy.in_fn.num_features}")
    
    # Calculate observation dimension for replay buffer
    # Each agent sees: original_obs + delay_buffer_size * action_dim
    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [env.observation_space[i].shape[0] + env.action_space[i].shape[0] * delay_buffer_size 
         for i in range(maddpg.nagents)],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
         for acsp in env.action_space]
    )
    
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        
        obs = env.reset()
        
        print(f"[DEBUG] After reset, obs shape: {obs.shape}")
        for i, o in enumerate(obs[0]):
            print(f"[DEBUG] obs[0][{i}] shape: {o.shape}, dtype: {o.dtype}")
        
        if USE_CUDA:
            maddpg.prep_rollouts(device='gpu')
        else:
            maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # Initialize action buffers for each environment
        # Structure: last_agent_actions[env_idx][agent_idx] = [newest, ..., oldest]
        last_agent_actions = []
        for env_idx in range(config.n_rollout_threads):
            env_agent_buffers = []
            for agent_idx in range(maddpg.nagents):
                # Initialize with zeros, size = delay_buffer_size
                zero_actions = [np.zeros(env.action_space[agent_idx].shape[0]) 
                               for _ in range(delay_buffer_size)]
                env_agent_buffers.append(zero_actions)
            last_agent_actions.append(env_agent_buffers)
        
        print(f"[DEBUG] Action buffer initialized with size {delay_buffer_size} for each agent")
        
        # Append action history to observations for ALL environments
        for env_idx in range(config.n_rollout_threads):
            for a_i in range(len(obs[env_idx])):
                agent_obs = obs[env_idx][a_i]
                # Append all actions in buffer (from newest to oldest)
                for action_idx in range(delay_buffer_size):
                    agent_obs = np.append(agent_obs, last_agent_actions[env_idx][a_i][action_idx])
                obs[env_idx, a_i] = agent_obs
        
        print(f"[DEBUG] After appending action history:")
        for env_idx in range(min(2, config.n_rollout_threads)):
            print(f"[DEBUG]   Env {env_idx}: agent shapes = {[obs[env_idx][a].shape for a in range(len(obs[env_idx]))]}")
        
        for et_i in range(config.episode_length):
            # Get observations for all agents
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            
            # Get actions from policies
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            #agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            # Prepare actions for each environment
            actions = []
            for env_idx in range(config.n_rollout_threads):
                # Get current actions for this environment
                current_actions = [agent_actions[a_i][env_idx] for a_i in range(maddpg.nagents)]
                
                # Compute virtual effective actions using the delay
                env_actions = []
                for agent_idx in range(maddpg.nagents):
                    virtual_action = compute_virtual_action(
                        last_agent_actions[env_idx][agent_idx], 
                        config.delay_step
                    )
                    env_actions.append(virtual_action)
                
                # Update action buffers: shift and add new action
                for agent_idx in range(maddpg.nagents):
                    # Remove oldest action
                    last_agent_actions[env_idx][agent_idx].pop()
                    # Add newest action at the beginning
                    last_agent_actions[env_idx][agent_idx].insert(0, current_actions[agent_idx])
                
                actions.append(env_actions)
            
            # Step environment with virtual effective actions
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Append action history to next observations
            for env_idx in range(config.n_rollout_threads):
                for a_i in range(len(next_obs[env_idx])):
                    agent_obs = next_obs[env_idx][a_i]
                    # Append all actions in buffer
                    for action_idx in range(delay_buffer_size):
                        agent_obs = np.append(agent_obs, last_agent_actions[env_idx][a_i][action_idx])
                    next_obs[env_idx, a_i] = agent_obs
            
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
                    for a_i in range(maddpg.nagents - 1):  # do not update the runner
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
    parser.add_argument("--delay_step", 
                        default=3.0, type=float,
                        help="Delay in time steps (can be non-integral, e.g., 2.5)")

    config = parser.parse_args()

    run(config)
