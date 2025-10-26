class MultiAgentEnvAdapter:
    def __init__(self, pettingzoo_env):
        self.env = pettingzoo_env
        obs_dict, _ = self.env.reset()
        self.agents = list(obs_dict.keys())


    def reset(self):
        obs_dict, _ = self.env.reset()
        obs_n = [obs_dict[a] for a in self.agents]
        return obs_n

    def step(self, action_n):
        # action_n: list of actions, one per agent, in agent order
        actions = {a: act for a, act in zip(self.agents, action_n)}
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        obs_n = [obs_dict[a] for a in self.agents]
        reward_n = [rewards[a] for a in self.agents]
        done_n = [terminations[a] or truncations[a] for a in self.agents]
        # Optionally, handle info_n to mimic old format
        info_n = {'n': [infos[a] for a in self.agents]}
        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    # Add other methods as needed to match old class (e.g., observation_space, action_space)
    @property
    def action_space(self):
        return [self.env.action_space(a) for a in self.agents]

    @property
    def observation_space(self):
        return [self.env.observation_space(a) for a in self.agents]


def make_env(scenario_name, discrete_action=False):
    from pettingzoo.mpe.simple_speaker_listener_v4 import parallel_env as ssl_env
    from pettingzoo.mpe.simple_spread_v3 import parallel_env as ss_env
    from pettingzoo.mpe.simple_reference_v3 import parallel_env as sr_env

    scenario_dict = {
        'simple_speaker_listener': ssl_env,
        'simple_spread': ss_env,
        'simple_reference': sr_env,
    }

    if scenario_name not in scenario_dict:
        raise ValueError(f"Scenario {scenario_name} not found in MPE2 environments")

    env = scenario_dict[scenario_name](
        max_cycles=25,
        continuous_actions=not discrete_action
    )
    env = MultiAgentEnvAdapter(env)
    return env
