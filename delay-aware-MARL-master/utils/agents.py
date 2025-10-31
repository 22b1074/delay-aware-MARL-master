import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import numpy as np
device = 'cuda'
class DDPGAgent(object):
    """
    Delay-aware wrapper for DDPG with action history observation augmentation
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, delay_step=3.0, use_sigmoid=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input (obs + action_history)
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
            hidden_dim (int): number of hidden dimensions
            lr (float): learning rate
            discrete_action (bool): whether action space is discrete
            delay_step (float): delay in timesteps (can be fractional)
            use_sigmoid (bool): use sigmoid for continuous actions (outputs [0,1])
        """
        self.discrete_action = discrete_action
        self.delay_step = delay_step
        self.use_sigmoid = use_sigmoid
        
        print(f"[DelayAwareDDPG] Initializing agent:")
        print(f"  Policy input dim: {num_in_pol}")
        print(f"  Policy output dim: {num_out_pol}")
        print(f"  Critic input dim: {num_in_critic}")
        print(f"  Delay step: {delay_step}")
        print(f"  Use sigmoid: {use_sigmoid}")
        
        # Policy network - uses sigmoid for continuous actions
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 use_sigmoid=use_sigmoid)
        
        # Critic network
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        
        # Target networks
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        use_sigmoid=use_sigmoid)
        
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if next(self.policy.parameters()).is_cuda:
            device = torch.device("cuda")
            obs = obs.to(device)
        else:
            device = torch.device("cpu")
            obs = obs.to(device)
        action = self.policy(obs)
        
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                noise = Variable(Tensor(self.exploration.noise()),
                                requires_grad=False).to(device)
                action = action + noise
            
            # Clamp to [0, 1] if using sigmoid, or [-1, 1] if using tanh
            if self.use_sigmoid:
                # Use epsilon = 1e-6 to stay strictly inside (0, 1)
                epsilon = 1e-6
                action = action.clamp(epsilon, 1.0 - epsilon)
            else:
                # For tanh: clamp to (-1 + eps, 1 - eps)
                epsilon = 1e-6
                action = action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        
        # Convert to numpy with FLOAT32 (not float64)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().astype(np.float32)  # ← CRITICAL: float32
        
        # EXTRA SAFETY: Ensure strictly within bounds and correct dtype
        if self.use_sigmoid:
            epsilon = 1e-6
            action = np.clip(action, epsilon, 1.0 - epsilon).astype(np.float32)
        else:
            epsilon = 1e-6
            action = np.clip(action, -1.0 + epsilon, 1.0 - epsilon).astype(np.float32)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
