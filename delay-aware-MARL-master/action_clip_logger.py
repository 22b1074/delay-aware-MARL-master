import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class ActionClipLogger(gym.Wrapper):
    """
    Wrapper that logs whenever actions are clipped to fit within action space bounds.
    Shows both the original action and what it was clipped to.
    """
    
    def __init__(self, env, log_frequency=100, verbose=True):
        """
        Args:
            env: The environment to wrap
            log_frequency: How often to log (every N steps). Set to 1 to log every step.
            verbose: If True, print detailed clipping info. If False, only count clipping events.
        """
        super().__init__(env)
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.step_count = 0
        self.clip_count = 0
        self.total_clip_magnitude = 0.0
        
    def step(self, action):
        """Step the environment and log any action clipping."""
        self.step_count += 1
        
        # Check if action needs clipping
        clipping_occurred = False
        original_action = np.array(action).copy()
        
        if isinstance(self.action_space, Box):
            # Check each dimension
            out_of_bounds = (action < self.action_space.low) | (action > self.action_space.high)
            
            if np.any(out_of_bounds):
                clipping_occurred = True
                self.clip_count += 1
                
                # Clip the action
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
                clip_magnitude = np.abs(original_action - clipped_action).sum()
                self.total_clip_magnitude += clip_magnitude
                
                # Log if verbose and within frequency
                if self.verbose and (self.step_count % self.log_frequency == 0 or self.step_count <= 10):
                    print(f"\n[ACTION CLIP] Step {self.step_count}")
                    print(f"  Original action:     {original_action}")
                    print(f"  Clipped action:      {clipped_action}")
                    print(f"  Action space bounds: [{self.action_space.low}, {self.action_space.high}]")
                    print(f"  Difference:          {original_action - clipped_action}")
                    print(f"  Clip magnitude:      {clip_magnitude:.6f}")
                    print(f"  Total clips so far:  {self.clip_count}/{self.step_count}")
                    
                    # Show which dimensions were clipped
                    for i, oob in enumerate(out_of_bounds):
                        if oob:
                            if original_action[i] < self.action_space.low[i]:
                                print(f"    Dim {i}: {original_action[i]:.6f} < {self.action_space.low[i]:.6f} (clipped up)")
                            else:
                                print(f"    Dim {i}: {original_action[i]:.6f} > {self.action_space.high[i]:.6f} (clipped down)")
                
                # Use clipped action
                action = clipped_action
        
        # Step with the (potentially clipped) action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add clipping info to info dict
        info['action_clipped'] = clipping_occurred
        if clipping_occurred:
            info['original_action'] = original_action
            info['clipped_action'] = action
        
        return obs, reward, terminated, truncated, info
    
    def get_clip_statistics(self):
        """Return statistics about clipping."""
        if self.step_count == 0:
            return {
                'total_steps': 0,
                'total_clips': 0,
                'clip_rate': 0.0,
                'avg_clip_magnitude': 0.0
            }
        
        return {
            'total_steps': self.step_count,
            'total_clips': self.clip_count,
            'clip_rate': self.clip_count / self.step_count,
            'avg_clip_magnitude': self.total_clip_magnitude / max(1, self.clip_count)
        }
    
    def print_statistics(self):
        """Print clipping statistics."""
        stats = self.get_clip_statistics()
        print("\n" + "="*60)
        print("ACTION CLIPPING STATISTICS")
        print("="*60)
        print(f"Total steps:              {stats['total_steps']}")
        print(f"Actions clipped:          {stats['total_clips']}")
        print(f"Clip rate:                {stats['clip_rate']*100:.2f}%")
        print(f"Avg clip magnitude:       {stats['avg_clip_magnitude']:.6f}")
        print("="*60 + "\n")


class MultiAgentActionClipLogger:
    """
    Wrapper for multi-agent environments that logs action clipping per agent.
    Works with vectorized environments.
    """
    
    def __init__(self, env, log_frequency=100, verbose=True):
        self.env = env
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.step_count = 0
        self.clip_counts = {}  # Per agent
        self.total_clip_magnitudes = {}  # Per agent
        
        # Initialize counters for each agent
        if hasattr(env, 'action_space') and isinstance(env.action_space, list):
            self.n_agents = len(env.action_space)
            for i in range(self.n_agents):
                self.clip_counts[i] = 0
                self.total_clip_magnitudes[i] = 0.0
    
    def check_and_log_clipping(self, actions, step_num=None):
        """
        Check if actions need clipping and log the information.
        
        Args:
            actions: List of actions for each environment (for vectorized envs)
                    Each element should be a list of actions per agent
            step_num: Current step number (optional)
        
        Returns:
            clipped_actions: Actions clipped to valid range, preserving structure
        """
        if step_num is not None:
            self.step_count = step_num
        else:
            self.step_count += 1
        
        # Handle single environment or vectorized
        if not isinstance(actions, list):
            actions = [actions]
        
        clipped_actions = []
        
        for env_idx, env_actions in enumerate(actions):
            env_clipped = []
            
            for agent_idx, action in enumerate(env_actions):
                # Ensure action is a numpy array
                original_action = np.asarray(action, dtype=self.env.action_space[agent_idx].dtype)
                action_space = self.env.action_space[agent_idx]
                
                if isinstance(action_space, Box):
                    # Check if clipping needed
                    out_of_bounds = (original_action < action_space.low) | (original_action > action_space.high)
                    
                    if np.any(out_of_bounds):
                        self.clip_counts[agent_idx] = self.clip_counts.get(agent_idx, 0) + 1
                        
                        # Clip the action
                        clipped_action = np.clip(original_action, action_space.low, action_space.high)
                        clip_magnitude = np.abs(original_action - clipped_action).sum()
                        self.total_clip_magnitudes[agent_idx] = self.total_clip_magnitudes.get(agent_idx, 0) + clip_magnitude
                        
                        # Log if verbose
                        should_log = (self.step_count % self.log_frequency == 0 or 
                                     self.step_count <= 10)
                        
                        if self.verbose and should_log:
                            print(f"\n[ACTION CLIP] Step {self.step_count}, Env {env_idx}, Agent {agent_idx}")
                            print(f"  Original:  {original_action}")
                            print(f"  Clipped:   {clipped_action}")
                            print(f"  Bounds:    [{action_space.low}, {action_space.high}]")
                            print(f"  Diff:      {original_action - clipped_action}")
                            print(f"  Magnitude: {clip_magnitude:.6f}")
                            
                            # Show which dimensions were clipped
                            for i, oob in enumerate(out_of_bounds):
                                if oob:
                                    if original_action[i] < action_space.low[i]:
                                        print(f"    Dim {i}: {original_action[i]:.6f} < LOW {action_space.low[i]:.6f}")
                                    else:
                                        print(f"    Dim {i}: {original_action[i]:.6f} > HIGH {action_space.high[i]:.6f}")
                        
                        # Ensure clipped action is correct dtype and shape
                        clipped_action = np.asarray(clipped_action, dtype=action_space.dtype)
                        env_clipped.append(clipped_action)
                    else:
                        # No clipping needed, but ensure correct dtype and shape
                        env_clipped.append(np.asarray(original_action, dtype=action_space.dtype))
                else:
                    # Not a Box space, pass through
                    env_clipped.append(np.asarray(action, dtype=action_space.dtype))
            
            clipped_actions.append(env_clipped)
        
        # Return single list if input was single environment
        return clipped_actions if len(clipped_actions) > 1 else clipped_actions[0]
    
    def print_statistics(self):
        """Print per-agent clipping statistics."""
        print("\n" + "="*70)
        print("MULTI-AGENT ACTION CLIPPING STATISTICS")
        print("="*70)
        print(f"Total steps: {self.step_count}")
        print()
        
        for agent_idx in sorted(self.clip_counts.keys()):
            clips = self.clip_counts[agent_idx]
            clip_rate = clips / max(1, self.step_count) * 100
            avg_mag = self.total_clip_magnitudes[agent_idx] / max(1, clips)
            
            print(f"Agent {agent_idx}:")
            print(f"  Clips:          {clips}")
            print(f"  Clip rate:      {clip_rate:.2f}%")
            print(f"  Avg magnitude:  {avg_mag:.6f}")
            print()
        print("="*70 + "\n")
