from utils.networks import MLPNetwork
import torch
import torch.nn.functional as F

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, target policy, critic, target critic)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, use_sigmoid=True):  # Add use_sigmoid param
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
            use_sigmoid (bool): Use sigmoid activation for continuous actions
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 use_sigmoid=use_sigmoid)  # Pass it here
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        use_sigmoid=use_sigmoid)  # And here
        
        self.critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim,
                                        constrain_out=False)

        # Copy parameters from policy to target_policy
        for target_param, param in zip(self.target_policy.parameters(),
                                        self.policy.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_critic.parameters(),
                                        self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.exploration_noise = OUNoise(num_out_pol)
        self.discrete_action = discrete_action

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:
            if explore:
                # Add noise - but now action is in [0, 1] not [-1, 1]
                # So we need to scale noise appropriately
                noise = Variable(torch.Tensor(self.exploration_noise.noise()),
                                requires_grad=False)
                # Scale noise to [0, 1] range
                noise = (noise + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
                action = action + noise
                # Clip to valid range
                action = torch.clamp(action, 0.0, 1.0)
            else:
                # Clip to ensure in valid range
                action = torch.clamp(action, 0.0, 1.0)
        
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
