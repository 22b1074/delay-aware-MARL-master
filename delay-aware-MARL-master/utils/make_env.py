# multiagent_env_adapter.py
import numpy as np
from pettingzoo.mpe.simple_speaker_listener_v4 import parallel_env as ssl_env
from pettingzoo.mpe.simple_spread_v3 import parallel_env as ss_env
from pettingzoo.mpe.simple_reference_v3 import parallel_env as sr_env
class MultiAgentEnvAdapter:
    """
    Adapter to convert PettingZoo parallel env (dict obs/action) to old MPE style list interface
    """
    def __init__(self, pettingzoo_env):
        self.env = pettingzoo_env
        obs_dict, _ = self.env.reset()
        self.agents = list(obs_dict.keys())
        self.n = len(self.agents)
        print(f"[DEBUG] Agents: {self.agents}, Total: {self.n}")
    def reset(self, **kwargs):
        obs_dict, _ = self.env.reset(**kwargs)
        obs_n = [obs_dict[a] for a in self.agents]
        
        # ADD THESE DEBUG PRINTS HERE:
        print(f"[DEBUG] obs_n types: {[type(o) for o in obs_n]}")
        print(f"[DEBUG] obs_n shapes: {[o.shape if hasattr(o, 'shape') else 'no shape' for o in obs_n]}")
        print(f"[DEBUG] obs_n dtypes: {[o.dtype if hasattr(o, 'dtype') else 'no dtype' for o in obs_n]}")
        
        return obs_n

    def step(self, action_n):
        """
        action_n: list of actions for each agent in self.agents order
        Returns: obs_n, reward_n, done_n, info_n
        """
        # map list -> dict for PettingZoo env
        actions = {a: act for a, act in zip(self.agents, action_n)}
        obs_dict, rewards_dict, terminations, truncations, infos_dict = self.env.step(actions)

        obs_n = [obs_dict[a] for a in self.agents]
        reward_n = [rewards_dict[a] for a in self.agents]
        done_n = [terminations[a] or truncations[a] for a in self.agents]
        info_n = {'n': [infos_dict[a] for a in self.agents]}

        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    @property
    def action_space(self):
        spaces = [self.env.action_space(a) for a in self.agents]
        for i, sp in enumerate(spaces):
            print(f"[DEBUG] Agent {self.agents[i]} Action space: {sp}, Type: {type(sp)}")
        return spaces
    @property
    def observation_space(self):
        spaces = [self.env.observation_space(a) for a in self.agents]
        for i, sp in enumerate(spaces):
            print(f"[DEBUG] Agent {self.agents[i]} Observation space: {sp}, Type: {type(sp)}")
        return spaces


def make_env(scenario_name, discrete_action=False):
    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
    }

    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found in MPE2 environments")

    env = scenario_dict[scenario_name](
        max_cycles=25,
        continuous_actions=not discrete_action,
        render_mode=render_mode
    )
    env = MultiAgentEnvAdapter(env)
    return env
