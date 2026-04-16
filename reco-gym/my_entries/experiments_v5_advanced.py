"""
RecoGym V5 - 算法增强与状态增强实验
=====================================
目标：
1. 尝试算法增强：Actor-Critic、PPO
2. 尝试状态增强：用户序列、上下文、正负反馈

说明：
- 继续使用 NumPy + RecoGym，不引入额外深度学习框架
- 用线性策略/价值函数 + 手工状态特征，验证方法思想
- 输出两组实验：
  A. 算法增强对比（Popularity / Actor-Critic / PPO）
  B. 状态增强消融（Simple State vs Enhanced State）
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print('=' * 72)
print('RecoGym V5 - 算法增强与状态增强实验')
print('=' * 72)

CONFIG = {
    'random_seed': 42,
    'num_products': 20,
    'num_offline_users': 150,
    'num_online_users': 80,
    'state_window': 5,
    'gamma': 0.95,
}

config_dict = {
    **env_1_args,
    'random_seed': CONFIG['random_seed'],
    'num_products': CONFIG['num_products'],
    'K': 6,
    'sigma_omega': 0.25,
}


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z) + 1e-12)


class StateFeatureBuilder:
    """构建两种状态：
    1. simple: 仅使用最近浏览物品的频次
    2. enhanced: 用户序列 + 上下文 + 正负反馈
    """

    def __init__(self, num_products: int, window: int = 5):
        self.num_products = num_products
        self.window = window
        self.positive_history: List[int] = []
        self.negative_history: List[int] = []
        self.last_action: int | None = None
        self.last_reward: float = 0.0

    def reset(self) -> None:
        self.positive_history.clear()
        self.negative_history.clear()
        self.last_action = None
        self.last_reward = 0.0

    def update_feedback(self, action: int | None, reward: float | None) -> None:
        if action is None or reward is None:
            return
        self.last_action = int(action)
        self.last_reward = float(reward)
        if reward > 0:
            self.positive_history.append(int(action))
            self.positive_history = self.positive_history[-self.window:]
        else:
            self.negative_history.append(int(action))
            self.negative_history = self.negative_history[-self.window:]

    def _recent_session_items(self, observation) -> List[int]:
        if observation is None:
            return []
        sessions = observation.sessions()
        if sessions is None:
            return []
        items = [s['v'] for s in sessions if 'v' in s]
        return items[-self.window:]

    def simple_state(self, observation) -> np.ndarray:
        state = np.zeros(self.num_products + 1)
        recent_items = self._recent_session_items(observation)
        for item in recent_items:
            state[item] += 1.0
        if len(recent_items) > 0:
            state[:self.num_products] /= len(recent_items)
        state[-1] = len(recent_items) / max(1, self.window)
        return state

    def enhanced_state(self, observation) -> np.ndarray:
        recent_items = self._recent_session_items(observation)
        seq_feature = np.zeros(self.num_products)
        pos_feature = np.zeros(self.num_products)
        neg_feature = np.zeros(self.num_products)

        for idx, item in enumerate(recent_items):
            seq_feature[item] += (idx + 1) / max(1, len(recent_items))
        if np.sum(seq_feature) > 0:
            seq_feature /= np.sum(seq_feature)

        for item in self.positive_history:
            pos_feature[item] += 1.0
        for item in self.negative_history:
            neg_feature[item] += 1.0

        if np.sum(pos_feature) > 0:
            pos_feature /= np.sum(pos_feature)
        if np.sum(neg_feature) > 0:
            neg_feature /= np.sum(neg_feature)

        context = np.zeros(4)
        if observation is not None:
            t = float(observation.context().time())
            user = float(observation.context().user())
            context[0] = (t % 24.0) / 24.0
            context[1] = (t % 7.0) / 7.0
            context[2] = len(recent_items) / max(1, self.window)
            context[3] = 1.0 if user % 2 == 0 else 0.0

        feedback = np.array([
            1.0 if self.last_reward > 0 else 0.0,
            1.0 if (self.last_action is not None and self.last_reward <= 0) else 0.0,
        ])

        return np.concatenate([seq_feature, pos_feature, neg_feature, context, feedback])


class PopularityAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.product_views = np.zeros(config.num_products)

    def train(self, observation, action, reward, done=False):
        if observation is not None:
            for session in observation.sessions():
                self.product_views[session['v']] += 1

    def act(self, observation, reward, done):
        if np.sum(self.product_views) == 0:
            action = np.random.randint(self.config.num_products)
        else:
            action = int(np.argmax(self.product_views))
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.config.num_products),
        }

    def reset(self):
        pass


class LinearActorCriticAgent(Agent):
    def __init__(self, config, use_enhanced_state=True, actor_lr=0.05, critic_lr=0.1, gamma=0.95, entropy_coef=0.01):
        super().__init__(config)
        self.num_products = config.num_products
        self.use_enhanced_state = use_enhanced_state
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.feature_builder = StateFeatureBuilder(self.num_products, window=CONFIG['state_window'])
        self.state_dim = (self.num_products * 3 + 6) if use_enhanced_state else (self.num_products + 1)

        self.actor_w = np.zeros((self.num_products, self.state_dim))
        self.critic_w = np.zeros(self.state_dim)

        self.prev_state: np.ndarray | None = None
        self.prev_action: int | None = None
        self.prev_prob: float = 0.0

    def reset(self):
        self.feature_builder.reset()
        self.prev_state = None
        self.prev_action = None
        self.prev_prob = 0.0

    def _state_vector(self, observation) -> np.ndarray:
        if self.use_enhanced_state:
            return self.feature_builder.enhanced_state(observation)
        return self.feature_builder.simple_state(observation)

    def _policy(self, state: np.ndarray) -> np.ndarray:
        logits = self.actor_w @ state
        return softmax(logits)

    def _value(self, state: np.ndarray) -> float:
        return float(np.dot(self.critic_w, state))

    def act(self, observation, reward, done):
        state = self._state_vector(observation)
        policy = self._policy(state)
        action = int(np.random.choice(self.num_products, p=policy))
        self.prev_state = state.copy()
        self.prev_action = action
        self.prev_prob = float(policy[action])
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': float(policy[action]),
            'ps-a': policy,
        }

    def train(self, observation, action, reward, done=False):
        if action is None:
            return

        self.feature_builder.update_feedback(action['a'], reward)
        next_state = self._state_vector(observation)

        if self.prev_state is None or self.prev_action is None:
            self.prev_state = next_state.copy()
            self.prev_action = action['a']
            return

        v_s = self._value(self.prev_state)
        v_next = 0.0 if done else self._value(next_state)
        delta = float(reward) + self.gamma * v_next - v_s

        self.critic_w += self.critic_lr * delta * self.prev_state

        policy = self._policy(self.prev_state)
        grad_log_pi = -np.outer(policy, self.prev_state)
        grad_log_pi[self.prev_action] += self.prev_state
        entropy_grad = -np.outer(policy * np.log(policy + 1e-12), self.prev_state)
        self.actor_w += self.actor_lr * (delta * grad_log_pi + self.entropy_coef * entropy_grad)

        self.prev_state = next_state.copy()
        self.prev_action = action['a']


class LinearPPOAgent(Agent):
    def __init__(self, config, use_enhanced_state=True, actor_lr=0.03, critic_lr=0.08, gamma=0.95, clip_eps=0.2, update_epochs=3):
        super().__init__(config)
        self.num_products = config.num_products
        self.use_enhanced_state = use_enhanced_state
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs

        self.feature_builder = StateFeatureBuilder(self.num_products, window=CONFIG['state_window'])
        self.state_dim = (self.num_products * 3 + 6) if use_enhanced_state else (self.num_products + 1)

        self.actor_w = np.zeros((self.num_products, self.state_dim))
        self.critic_w = np.zeros(self.state_dim)
        self.trajectory: List[Dict[str, np.ndarray | float | int]] = []

    def reset(self):
        self.feature_builder.reset()
        self.trajectory = []

    def _state_vector(self, observation) -> np.ndarray:
        if self.use_enhanced_state:
            return self.feature_builder.enhanced_state(observation)
        return self.feature_builder.simple_state(observation)

    def _policy(self, state: np.ndarray) -> np.ndarray:
        return softmax(self.actor_w @ state)

    def _value(self, state: np.ndarray) -> float:
        return float(np.dot(self.critic_w, state))

    def act(self, observation, reward, done):
        state = self._state_vector(observation)
        policy = self._policy(state)
        action = int(np.random.choice(self.num_products, p=policy))
        value = self._value(state)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': float(policy[action]),
            'ps-a': policy,
            'state': state,
            'value': value,
        }

    def train(self, observation, action, reward, done=False):
        if action is None:
            return

        # RecoGym 的 step_offline() 返回的是日志动作，不包含当前策略生成的 state/value。
        # PPO 这里主要在在线交互阶段做 on-policy 更新；离线阶段只同步反馈状态，不写入轨迹。
        if 'state' not in action or 'value' not in action:
            if isinstance(action, dict) and 'a' in action:
                self.feature_builder.update_feedback(action['a'], reward)
            return

        next_state = self._state_vector(observation)
        self.trajectory.append({
            'state': np.array(action['state'], copy=True),
            'action': int(action['a']),
            'reward': float(reward),
            'old_prob': float(max(action['ps'], 1e-8)),
            'value': float(action['value']),
            'next_value': 0.0 if done else self._value(next_state),
            'done': float(done),
        })

        self.feature_builder.update_feedback(action['a'], reward)

        if done:
            self.update()
            self.trajectory = []

    def update(self):
        if not self.trajectory:
            return

        rewards = [step['reward'] for step in self.trajectory]
        values = [step['value'] for step in self.trajectory]
        dones = [step['done'] for step in self.trajectory]

        returns = []
        advantages = []
        G = 0.0
        for idx in reversed(range(len(rewards))):
            G = rewards[idx] + self.gamma * G * (1.0 - dones[idx])
            returns.insert(0, G)
            advantages.insert(0, G - values[idx])

        returns = np.array(returns)
        advantages = np.array(advantages)
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(self.update_epochs):
            for step, ret, adv in zip(self.trajectory, returns, advantages):
                state = step['state']
                action = int(step['action'])
                old_prob = float(step['old_prob'])

                policy = self._policy(state)
                new_prob = float(max(policy[action], 1e-8))
                ratio = new_prob / old_prob
                clipped_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                if ratio <= clipped_ratio:
                    coeff = ratio
                else:
                    coeff = clipped_ratio

                grad_log_pi = -np.outer(policy, state)
                grad_log_pi[action] += state
                self.actor_w += self.actor_lr * coeff * adv * grad_log_pi

                value = self._value(state)
                self.critic_w += self.critic_lr * (ret - value) * state


# ============================================
# 训练与评估
# ============================================

def train_agent_offline(env, agent, num_users):
    uid = 0
    for _ in range(num_users):
        env.reset(uid)
        uid += 1
        if hasattr(agent, 'reset'):
            agent.reset()

        observation, _, done, _ = env.step(None)
        agent.train(observation, None, None, True)

        while not done:
            old_observation = observation
            action, observation, reward, done, _ = env.step_offline(old_observation, 0, done)
            agent.train(observation, action, reward, done)


def evaluate_agent_online(env, agent, num_users):
    uid = 2000
    total_clicks = 0
    total_steps = 0

    for _ in range(num_users):
        env.reset(uid)
        uid += 1
        agent.reset()
        observation, _, done, _ = env.step(None)

        while not done:
            action = agent.act(observation, 0, done)
            observation, reward, done, _ = env.step(action['a'])
            agent.train(observation, action, reward, done)
            total_clicks += reward
            total_steps += 1

        if hasattr(agent, 'update') and isinstance(agent, LinearPPOAgent):
            agent.update()
            agent.trajectory = []

    return total_clicks / total_steps if total_steps > 0 else 0.0


def build_env(seed_offset=0):
    env = RecoEnv1()
    env.init_gym({**config_dict, 'random_seed': CONFIG['random_seed'] + seed_offset})
    return env


print(f"\n实验配置: {CONFIG['num_products']}物品, {CONFIG['num_offline_users']}离线用户, {CONFIG['num_online_users']}在线用户")

base_env = build_env(0)
config = base_env.config

# ============================================
# 实验A：算法增强
# ============================================
print('\n' + '=' * 72)
print('实验A：算法增强（Popularity vs Actor-Critic vs PPO）')
print('=' * 72)

algo_agents = {
    'Popularity': lambda: PopularityAgent(config),
    'Actor-Critic': lambda: LinearActorCriticAgent(config, use_enhanced_state=True, gamma=CONFIG['gamma']),
    'PPO': lambda: LinearPPOAgent(config, use_enhanced_state=True, gamma=CONFIG['gamma']),
}

algo_results: Dict[str, float] = {}
for idx, (name, ctor) in enumerate(algo_agents.items(), start=1):
    print(f'\n{name}:')
    env = build_env(idx * 10)
    agent = ctor()
    train_agent_offline(env, agent, CONFIG['num_offline_users'])
    if hasattr(agent, 'update') and isinstance(agent, LinearPPOAgent):
        agent.update()
    ctr = evaluate_agent_online(env, agent, CONFIG['num_online_users'])
    algo_results[name] = ctr
    print(f'CTR = {ctr:.4f}')

# ============================================
# 实验B：状态增强消融
# ============================================
print('\n' + '=' * 72)
print('实验B：状态增强消融（Simple State vs Enhanced State）')
print('=' * 72)

state_agents = {
    'AC-Simple': lambda: LinearActorCriticAgent(config, use_enhanced_state=False, gamma=CONFIG['gamma']),
    'AC-Enhanced': lambda: LinearActorCriticAgent(config, use_enhanced_state=True, gamma=CONFIG['gamma']),
    'PPO-Simple': lambda: LinearPPOAgent(config, use_enhanced_state=False, gamma=CONFIG['gamma']),
    'PPO-Enhanced': lambda: LinearPPOAgent(config, use_enhanced_state=True, gamma=CONFIG['gamma']),
}

state_results: Dict[str, float] = {}
for idx, (name, ctor) in enumerate(state_agents.items(), start=1):
    print(f'\n{name}:')
    env = build_env(100 + idx * 10)
    agent = ctor()
    train_agent_offline(env, agent, CONFIG['num_offline_users'])
    if hasattr(agent, 'update') and isinstance(agent, LinearPPOAgent):
        agent.update()
    ctr = evaluate_agent_online(env, agent, CONFIG['num_online_users'])
    state_results[name] = ctr
    print(f'CTR = {ctr:.4f}')

# ============================================
# 可视化
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
labels_a = list(algo_results.keys())
values_a = list(algo_results.values())
colors_a = ['#3498db', '#2ecc71', '#9b59b6']
bars_a = ax1.bar(labels_a, values_a, color=colors_a, alpha=0.85)
ax1.set_title('算法增强实验')
ax1.set_ylabel('CTR')
ax1.set_ylim(0, max(values_a) * 1.25 if max(values_a) > 0 else 0.1)
for bar, value in zip(bars_a, values_a):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{value:.4f}', ha='center', va='bottom')

ax2 = axes[1]
labels_b = list(state_results.keys())
values_b = list(state_results.values())
colors_b = ['#95a5a6', '#27ae60', '#bdc3c7', '#8e44ad']
bars_b = ax2.bar(labels_b, values_b, color=colors_b, alpha=0.85)
ax2.set_title('状态增强消融实验')
ax2.set_ylabel('CTR')
ax2.set_ylim(0, max(values_b) * 1.25 if max(values_b) > 0 else 0.1)
for bar, value in zip(bars_b, values_b):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
out_path = PROJECT_ROOT / 'my_entries' / 'results_v5_advanced.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n图表已保存: {out_path.name}')

print('\n' + '=' * 72)
print('实验结果汇总')
print('=' * 72)
print('\n[算法增强]')
for name, value in sorted(algo_results.items(), key=lambda x: x[1], reverse=True):
    print(f'  {name:15s}: {value:.4f}')
print('\n[状态增强]')
for name, value in sorted(state_results.items(), key=lambda x: x[1], reverse=True):
    print(f'  {name:15s}: {value:.4f}')
print('=' * 72)
