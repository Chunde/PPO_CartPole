# %% [markdown]
# # Play PPO with cart pole for fun

# %%
import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as tc
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

sns.set_theme()
DEVICE = "cpu"


# %%
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_space_size)
        )

        self.value_layers = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        return policy_logits

    def forward(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value


# %% [markdown]
# ## Trainer


# %%
class PPOTrainer:
    def __init__(
        self,
        actor_critic: ActorCriticNetwork,
        ppo_clip_value=0.2,
        target_kl_div=0.01,
        max_policy_training_iterations=80,
        value_training_iterations=80,
        policy_lr=3e-4,
        value_lr=1e-2,
    ):
        self.ac = actor_critic
        self.ppo_clip_value = ppo_clip_value
        self.target_kl_div = target_kl_div
        self.max_policy_training_iterations = max_policy_training_iterations
        self.value_training_iterations = value_training_iterations

        policy_params = list(self.ac.shared_layers.parameters()) + list(
            self.ac.policy_layers.parameters()
        )
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)
        value_params = list(self.ac.shared_layers.parameters()) + list(
            self.ac.value_layers.parameters()
        )
        self.value_optim = optim.Adam(value_params, value_lr)

    def train_policy(self, obs, actions, old_log_probs, general_advantage_estimation):

        for _ in range(self.max_policy_training_iterations):
            self.policy_optim.zero_grad()
            new_logits = self.ac.policy(obs)
            new_logits = Categorical(logits=new_logits)
            new_logits_probs = new_logits.log_prob(actions)
            policy_ratio = tc.exp(new_logits_probs - old_log_probs)

            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_value, 1 + self.ppo_clip_value
            )
            policy_loss = (
                tc.min(policy_ratio, clipped_ratio) * general_advantage_estimation
            )

            policy_loss.backward()
            self.policy_optim.step()

            kl_divergence = (old_log_probs - new_logits).mean()
            if kl_divergence >= self.target_kl_div:
                break

    def train_value(self, obs, returns):
        for _ in range(self.value_training_iterations):
            self.value_optim.zero_grad()
            values = self.ac.value(obs)

            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()
            value_loss.backward()
            self.value_optim.step()


# %% [markdown]
# # Utility Functions

# %%
data = [1, 2, 3, 5, 6]
data[::-1]


# %%
def discount_reward(rewards, gamma=0.99):
    new_rewards = [float(rewards[-1])]

    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_general_advantage_estimates(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [
        reward + gamma * next_value - value
        for reward, value, next_value in zip(rewards, values, next_values)
    ]
    general_advantage_estimates = [deltas[-1]]

    for i in reversed(range(len(deltas) - 1)):
        general_advantage_estimates.append(
            deltas[i] + decay * gamma * general_advantage_estimates[-1]
        )

    return np.array(general_advantage_estimates[::-1])


# %% [markdown]
# ## Rollout


# %%
def rollout(model, env, max_steps=1000):

    # training data

    train_data = [
        [],
        [],
        [],
        [],
        [],
    ]  # save observation(0), action(1), reward(2), values(3) and action_log_probs
    obs, _ = env.reset()

    ep_reward = 0

    for _ in range(max_steps):
        logits, val = model(tc.tensor(obs, dtype=tc.float32, device=DEVICE))

        action_distribution = Categorical(logits=logits)
        action = action_distribution.sample()
        action_log_probability = action_distribution.log_prob(action).item()
        next_obs, reward, done, _, _ = env.step(action.item())

        # save the training data
        # here we pack all training data into one item and append it to the lists
        for i, item in enumerate(
            (
                obs,
                action.detach().numpy(),
                reward,
                val.detach().numpy()[0],
                action_log_probability,
            )
        ):
            train_data[i].append(item)

        obs = next_obs
        ep_reward += reward
        if done:
            break
    train_data = [np.asarray(x) for x in train_data]
    train_data[3] = calculate_general_advantage_estimates(train_data[2], train_data[3])
    return train_data, ep_reward


# %% [markdown]
# ## Create environment

# %%
env = gym.make("CartPole-v0")
model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)
model = model.to(DEVICE)
train_data, reward = rollout(model, env)

# %% [markdown]
# # Training Loop

# %% [markdown]
# * Setup PPO parameters

# %%
n_episodes = 200
print_freq = 20

ppo = PPOTrainer(
    model,
    policy_lr=3e-4,
    value_lr=1e-3,
    target_kl_div=0.02,
    max_policy_training_iterations=40,
    value_training_iterations=40,
)

# %%
ep_rewards = []

for episode_index in range(n_episodes):
    train_data, reward = rollout(model=model, env=env)
    ep_rewards.append(reward)

    # let us shuffle training data
    permute_indics = np.random.permutation(len(train_data[0]))
    permuted_obs = tc.tensor(
        train_data[0][permute_indics], dtype=tc.float32, device=DEVICE
    )
    permuted_values = tc.tensor(
        train_data[3][permute_indics], dtype=tc.float32, device=DEVICE
    )
    permuted_act = tc.tensor(
        train_data[1][permute_indics], dtype=tc.float32, device=DEVICE
    )
    permuted_act_log_probs = tc.tensor(
        train_data[4][permute_indics], dtype=tc.float32, device=DEVICE
    )

    returns = discount_reward(train_data[2])[permute_indics]
    returns = tc.tensor(returns, dtype=tc.float32, device=DEVICE)

    ppo.train_policy(
        permuted_obs, permuted_act, permuted_act_log_probs, permuted_values
    )
    ppo.train_value(permuted_obs, returns)
    if (episode_index - 1) % print_freq == 0:
        print(
            "Episode{} | Average Reward {:.1f}".format(
                episode_index + 1, np.mean(ep_rewards[-print_freq:])
            )
        )


# %%
