"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        self.envs = [env_fns[0]()]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_raw, rews, dones, infos = zip(*results)
        
        # Convert to 2D structure: [n_envs, n_agents] with each element being float32 array
        n_envs = len(obs_raw)
        n_agents = len(obs_raw[0])
        
        # Create proper 2D object array structure
        obs = np.empty((n_envs, n_agents), dtype=object)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                obs[env_idx, agent_idx] = np.asarray(obs_raw[env_idx][agent_idx], dtype=np.float32)
        
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        
        # Convert to 2D structure: [n_envs, n_agents] with each element being float32 array
        n_envs = len(results)
        n_agents = len(results[0])
        
        # Create proper 2D object array structure
        obs = np.empty((n_envs, n_agents), dtype=object)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                obs[env_idx, agent_idx] = np.asarray(results[env_idx][agent_idx], dtype=np.float32)
        
        return obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        results = [remote.recv() for remote in self.remotes]
        
        # Convert to 2D structure: [n_envs, n_agents] with each element being float32 array
        n_envs = len(results)
        n_agents = len(results[0])
        
        # Create proper 2D object array structure
        obs = np.empty((n_envs, n_agents), dtype=object)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                obs[env_idx, agent_idx] = np.asarray(results[env_idx][agent_idx], dtype=np.float32)
        
        return obs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        
        # Unpack results
        obs_raw, rews, dones, infos = zip(*results)
        
        # Convert to 2D structure: [n_envs, n_agents] with each element being float32 array
        n_envs = len(obs_raw)
        n_agents = len(obs_raw[0])
        
        # Create proper 2D object array structure
        obs = np.empty((n_envs, n_agents), dtype=object)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                obs[env_idx, agent_idx] = np.asarray(obs_raw[env_idx][agent_idx], dtype=np.float32)
        
        rews = np.array(rews)
        dones = np.array(dones)
        
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs_raw_reset = self.envs[i].reset()
                for agent_idx in range(n_agents):
                    obs[i, agent_idx] = np.asarray(obs_raw_reset[agent_idx], dtype=np.float32)
                self.ts[i] = 0
        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        
        # Convert to 2D structure: [n_envs, n_agents] with each element being float32 array
        n_envs = len(results)
        n_agents = len(results[0])
        
        # Create proper 2D object array structure
        obs = np.empty((n_envs, n_agents), dtype=object)
        for env_idx in range(n_envs):
            for agent_idx in range(n_agents):
                obs[env_idx, agent_idx] = np.asarray(results[env_idx][agent_idx], dtype=np.float32)
        
        return obs

    def close(self):
        return
