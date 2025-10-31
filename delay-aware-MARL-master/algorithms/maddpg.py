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
from action_clip_logger import MultiAgentActionClipLogger  # Add this import

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

def normalize_action_to_env(action, action_space):
    """
    Normalize action from model output range [-1, 1] to environment range [0, 1].
    
    Args:
        action: numpy array, model output in [-1, 1]
        action_space: gymnasium.spaces.Box with bounds
    
    Returns:
        Normalized action in [action_space.low, action_space.high]
    """
    # Clip to [-1, 1] first to handle exploration noise
    action = np.clip(action, -1.0, 1.0)
    # Map from [-1, 1] to [0, 1]
    action = (action + 1.0) / 2.0
    # Add small epsilon to avoid boundary issues
    epsilon = 1e-6
    action = np.clip(action, action_space.low + epsilon, action_space.high - epsilon)
    return action

def denormalize_action_for_buffer(action, action_space):
    """
    Convert action from environment range [0, 1] back to model training range [-1, 1].
    This is used when storing actions in replay buffer.
    
    Args:
        action: numpy array in [0, 1]
        action_space: gymnasium.spaces.Box
    
    Returns:
        Action in [-1, 1] range for training
    """
    # Clip to ensure within bounds
    epsilon = 1e-6
    action = np.clip(action, action_space.low + epsilon, action_space.high - epsilon)
    # Map from [0, 1] to [-1, 1]
    action = action * 2.0 - 1.0
    return action

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
    
    if f == 0 or I >= len(action_buffer) - 1:
        # Pure integral delay or at boundary
        return action_buffer[min(I, len(action_buffer) - 1)]
    else:
        # Non-integral delay: interpolate between two actions
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
    
    # Add action clipping logger
    clip_logger = MultiAgentActionClipLogger(env, log_frequency=50, verbose=True)
    
    print("\n[DEBUG] ========== ENVIRONMENT INFO ==========")
    print(f"[DEBUG] Delay step: {config.delay_step} (I={int(np.floor(config.delay_step))}, f={config.delay_step - int(np.floor(config.delay_step))})")
    print(f"[DEBUG] Action normalization: ENABLED (model [-1,1] -> env [0,1])")
    print(f"[DEBUG] Exploration noise: init={config.init_noise_scale}, final={config.final_noise_scale}")
    print(f"[DEBUG] Exploration episodes: {config.n_exploration_eps}")
    
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
        delay_step=config.delay_step
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
        
        if USE_CUDA:
            maddpg.prep_rollouts(device='gpu')
        else:
            maddpg.prep_rollouts(device='cpu')

        # Set exploration noise - this scales down over time
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        current_noise_scale = config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining
        maddpg.scale_noise(current_noise_scale)
        maddpg.reset_noise()
        
        if ep_i % 1000 == 0:
            print(f"[DEBUG] Episode {ep_i}: Exploration noise scale = {current_noise_scale:.4f}")

        # Initialize action buffers for each environment
        # Store actions in NORMALIZED [0, 1] range (as sent to environment)
        # Use 0.49 to avoid boundary issues (Box is half-open [0, 1))
        last_agent_actions = []
        for env_idx in range(config.n_rollout_threads):
            env_agent_buffers = []
            for agent_idx in range(maddpg.nagents):
                # Initialize with 0.49 (safe value in [0, 1) range)
                zero_actions = [np.ones(env.action_space[agent_idx].shape[0]) * 0.49 
                               for _ in range(delay_buffer_size)]
                env_agent_buffers.append(zero_actions)
            last_agent_actions.append(env_agent_buffers)
        
        if ep_i == 0:
            print(f"[DEBUG] Action buffer initialized with {delay_buffer_size} actions per agent")
            print(f"[DEBUG] Initial action value: 0.49 (safe within [0, 1) bounds)")
        
        # Append action history to observations for ALL environments
        # Actions stored in buffer are in [0, 1], need to convert to [-1, 1] for model
        for env_idx in range(config.n_rollout_threads):
            for a_i in range(len(obs[env_idx])):
                agent_obs = obs[env_idx][a_i]
                # Append all actions in buffer (from newest to oldest)
                # Convert from [0, 1] to [-1, 1] for model input
                for action_idx in range(delay_buffer_size):
                    normalized_action = denormalize_action_for_buffer(
                        last_agent_actions[env_idx][a_i][action_idx],
                        env.action_space[a_i]
                    )
                    agent_obs = np.append(agent_obs, normalized_action)
                obs[env_idx, a_i] = agent_obs
        
        for et_i in range(config.episode_length):
            # Get observations for all agents
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            
            # Get actions from policies with EXPLORATION NOISE
            # Output is in [-1, 1] range WITH noise already added
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions_raw = [ac.data.numpy() for ac in torch_agent_actions]
            
            # Normalize actions to environment range [0, 1]
            # This also clips to handle noise that pushed outside [-1, 1]
            agent_actions_normalized = []
            for agent_idx in range(maddpg.nagents):
                normalized = np.array([
                    normalize_action_to_env(agent_actions_raw[agent_idx][env_idx], 
                                           env.action_space[agent_idx])
                    for env_idx in range(config.n_rollout_threads)
                ])
                agent_actions_normalized.append(normalized)
            
            # Prepare actions for each environment
            actions = []
            for env_idx in range(config.n_rollout_threads):
                # Get current normalized actions for this environment
                current_actions = [agent_actions_normalized[a_i][env_idx] 
                                  for a_i in range(maddpg.nagents)]
                
                # Compute virtual effective actions using the delay
                env_actions = []
                for agent_idx in range(maddpg.nagents):
                    virtual_action = compute_virtual_action(
                        last_agent_actions[env_idx][agent_idx], 
                        config.delay_step
                    )
                    env_actions.append(virtual_action)
                
                # Update action buffers: shift and add new action (in [0, 1] range)
                for agent_idx in range(maddpg.nagents):
                    last_agent_actions[env_idx][agent_idx].pop()
                    last_agent_actions[env_idx][agent_idx].insert(0, current_actions[agent_idx])
                
                actions.append(env_actions)
            
            # Check and log clipping using the wrapper
            actions = clip_logger.check_and_log_clipping(actions, step_num=ep_i * config.episode_length + et_i)
            
            # Step environment with virtual effective actions (already in [0, 1])
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Append action history to next observations
            # Convert buffer actions from [0, 1] to [-1, 1] for model
            for env_idx in range(config.n_rollout_threads):
                for a_i in range(len(next_obs[env_idx])):
                    agent_obs = next_obs[env_idx][a_i]
                    for action_idx in range(delay_buffer_size):
                        normalized_action = denormalize_action_for_buffer(
                            last_agent_actions[env_idx][a_i][action_idx],
                            env.action_space[a_i]
                        )
                        agent_obs = np.append(agent_obs, normalized_action)
                    next_obs[env_idx, a_i] = agent_obs
            
            # Store RAW actions ([-1, 1] range) in replay buffer for training
            replay_buffer.push(obs, agent_actions_raw, rewards, next_obs, dones)
            
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
            print(f"Episode {ep_i+1}, Agent {a_i} total reward: {a_ep_rew:.4f}")
            logger.add_scalars('agent%i/mean_episode_rewards' % a_i, {'reward': a_ep_rew}, ep_i)
        
        # Print clipping statistics every 1000 episodes
        if (ep_i + 1) % 1000 == 0:
            clip_logger.print_statistics()

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    
    # Print final clipping statistics
    print("\n" + "="*70)
    print("FINAL TRAINING STATISTICS")
    print("="*70)
    clip_logger.print_statistics()
    
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
