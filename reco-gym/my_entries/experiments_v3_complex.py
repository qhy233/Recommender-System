"""
RecoGym 强化学习推荐系统实验 V3 - 复杂环境版
================================
目标：验证RL方法在复杂环境下的优势

环境增强：
1. 物品数量: 10 → 100
2. 潜在维度K: 5 → 20
3. 用户兴趣变化: σ=0.1 → σ=0.5
4. 增加训练数据量
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("RecoGym 强化学习推荐系统实验 V3 - 复杂环境版")
print("=" * 70)

# ============================================
# 复杂环境配置
# ============================================
COMPLEX_CONFIG = {
    'random_seed': 42,
    'num_products': 100,            # 增加物品数量
    'K': 20,                        # 增加潜在维度
    'sigma_omega_initial': 1.5,     # 用户初始兴趣多样性增加
    'sigma_omega': 0.5,             # 用户兴趣变化加快
    'sigma_mu_organic': 3.0,        # 物品流行度差异
    'num_offline_users': 500,       # 增加训练数据
    'num_online_users': 200,        # 增加评估数据
    'num_runs': 3,                  # 多次实验
}

# 创建复杂环境配置
complex_env_config = {
    **env_1_args,
    'random_seed': COMPLEX_CONFIG['random_seed'],
    'num_products': COMPLEX_CONFIG['num_products'],
    'K': COMPLEX_CONFIG['K'],
    'sigma_omega_initial': COMPLEX_CONFIG['sigma_omega_initial'],
    'sigma_omega': COMPLEX_CONFIG['sigma_omega'],
    'sigma_mu_organic': COMPLEX_CONFIG['sigma_mu_organic'],
}

print("\n环境配置对比:")
print("-" * 50)
print(f"  物品数量:      10 → {COMPLEX_CONFIG['num_products']}")
print(f"  潜在维度K:     5 → {COMPLEX_CONFIG['K']}")
print(f"  用户兴趣变化σ: 0.1 → {COMPLEX_CONFIG['sigma_omega']}")
print(f"  离线训练用户:  {COMPLEX_CONFIG['num_offline_users']}")
print(f"  在线评估用户:  {COMPLEX_CONFIG['num_online_users']}")
print(f"  重复实验次数:  {COMPLEX_CONFIG['num_runs']}")


# ============================================
# Agent定义
# ============================================

class PopularityAgent(Agent):
    """基于物品流行度的推荐Agent"""
    def __init__(self, config):
        super(PopularityAgent, self).__init__(config)
        self.product_views = np.zeros(config.num_products)
    
    def train(self, observation, action, reward, done=False):
        if observation is not None:
            for session in observation.sessions():
                self.product_views[session['v']] += 1
    
    def act(self, observation, reward, done):
        if np.sum(self.product_views) == 0:
            action = np.random.randint(self.config.num_products)
        else:
            action = np.argmax(self.product_views)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.config.num_products)
        }


class EpsilonGreedyAgent(Agent):
    """ε-greedy探索策略Agent"""
    def __init__(self, config, epsilon=0.1):
        super(EpsilonGreedyAgent, self).__init__(config)
        self.epsilon = epsilon
        self.product_clicks = np.zeros(config.num_products)
        self.product_recommendations = np.zeros(config.num_products)
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            self.product_recommendations[action['a']] += 1
            self.product_clicks[action['a']] += reward
    
    def act(self, observation, reward, done):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.config.num_products)
        else:
            click_rates = np.divide(
                self.product_clicks, 
                self.product_recommendations,
                out=np.zeros(self.config.num_products),
                where=self.product_recommendations != 0
            )
            action = np.argmax(click_rates)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.config.num_products)
        }
    
    def reset(self):
        self.product_clicks = np.zeros(self.config.num_products)
        self.product_recommendations = np.zeros(self.config.num_products)


class UCBAgent(Agent):
    """UCB探索策略Agent"""
    def __init__(self, config, c=2.0):
        super(UCBAgent, self).__init__(config)
        self.c = c
        self.num_products = config.num_products
        self.Q = np.zeros(self.num_products)
        self.N = np.zeros(self.num_products)
        self.total_count = 0
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            a = action['a']
            self.N[a] += 1
            self.Q[a] += (reward - self.Q[a]) / self.N[a]
            self.total_count += 1
    
    def act(self, observation, reward, done):
        if self.total_count < self.num_products:
            action = self.total_count
        else:
            confidence = self.c * np.sqrt(np.log(self.total_count + 1) / (self.N + 1e-8))
            ucb_values = self.Q + confidence
            action = np.argmax(ucb_values)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.num_products)
        }
    
    def reset(self):
        self.Q = np.zeros(self.num_products)
        self.N = np.zeros(self.num_products)
        self.total_count = 0


class ThompsonSamplingAgent(Agent):
    """Thompson Sampling探索策略Agent"""
    def __init__(self, config):
        super(ThompsonSamplingAgent, self).__init__(config)
        self.num_products = config.num_products
        self.alpha = np.ones(self.num_products)
        self.beta = np.ones(self.num_products)
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            a = action['a']
            if reward > 0:
                self.alpha[a] += 1
            else:
                self.beta[a] += 1
    
    def act(self, observation, reward, done):
        samples = np.random.beta(self.alpha, self.beta)
        action = np.argmax(samples)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': samples[action] / np.sum(samples),
            'ps-a': samples / np.sum(samples)
        }
    
    def reset(self):
        self.alpha = np.ones(self.num_products)
        self.beta = np.ones(self.num_products)


class PolicyGradientAgent(Agent):
    """策略梯度Agent (REINFORCE)"""
    def __init__(self, config, learning_rate=0.1):
        super(PolicyGradientAgent, self).__init__(config)
        self.lr = learning_rate
        self.num_products = config.num_products
        self.theta = np.zeros(self.num_products)
        self.trajectory = []
        self.temperature = 2.0
    
    def get_policy(self):
        exp_theta = np.exp(self.theta / self.temperature)
        return exp_theta / np.sum(exp_theta)
    
    def act(self, observation, reward, done):
        policy = self.get_policy()
        action = np.random.choice(self.num_products, p=policy)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': policy[action],
            'ps-a': policy
        }
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            self.trajectory.append({
                'action': action['a'],
                'reward': reward,
                'prob': action['ps']
            })
    
    def update(self):
        if len(self.trajectory) == 0:
            return
        
        rewards = [t['reward'] for t in self.trajectory]
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        
        if len(returns) > 1:
            baseline = np.mean(returns)
            advantages = returns - baseline
            std = np.std(advantages)
            if std > 1e-8:
                advantages = advantages / std
        else:
            advantages = returns
        
        for t, adv in zip(self.trajectory, advantages):
            action = t['action']
            prob = t['prob']
            self.theta[action] += self.lr * adv * (1 - prob)
        
        self.trajectory = []
    
    def reset(self):
        self.trajectory = []


class DQNAgent(Agent):
    """DQN Agent"""
    def __init__(self, config, learning_rate=0.1, gamma=0.99):
        super(DQNAgent, self).__init__(config)
        self.lr = learning_rate
        self.gamma = gamma
        self.num_products = config.num_products
        self.Q = np.zeros(self.num_products)
        self.replay_buffer = []
        self.buffer_size = 5000
        self.epsilon = 0.3
    
    def act(self, observation, reward, done):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_products)
        else:
            action = np.argmax(self.Q)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': 1.0,
            'ps-a': np.zeros(self.num_products)
        }
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            self.replay_buffer.append({
                'action': action['a'],
                'reward': reward
            })
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
    
    def update(self):
        if len(self.replay_buffer) < 64:
            return
        
        batch_size = min(128, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for idx in indices:
            exp = self.replay_buffer[idx]
            action = exp['action']
            reward = exp['reward']
            self.Q[action] += self.lr * (reward - self.Q[action])
        
        self.epsilon = max(0.05, self.epsilon * 0.995)
    
    def reset(self):
        self.replay_buffer = []


# ============================================
# 训练和评估函数
# ============================================

def train_agent_offline(env, agent, num_users):
    unique_user_id = 0
    for _ in range(num_users):
        env.reset(unique_user_id)
        unique_user_id += 1
        
        observation, _, done, _ = env.step(None)
        agent.train(observation, None, None, True)
        
        while not done:
            old_observation = observation
            action, observation, reward, done, _ = env.step_offline(old_observation, 0, done)
            agent.train(old_observation, action, reward, False)
        
        action, observation, reward, done, _ = env.step_offline(observation, 0, done)
        agent.train(observation, action, reward, True)


def evaluate_agent_online(env, agent, num_users):
    unique_user_id = 1000
    total_clicks = 0
    total_steps = 0
    
    for _ in range(num_users):
        env.reset(unique_user_id)
        unique_user_id += 1
        agent.reset()
        
        observation, _, done, _ = env.step(None)
        
        while not done:
            action = agent.act(observation, 0, done)
            observation, reward, done, _ = env.step(action['a'])
            agent.train(observation, action, reward, done)
            
            total_clicks += reward
            total_steps += 1
    
    return total_clicks / total_steps if total_steps > 0 else 0


def run_single_experiment(agent_class, config, env_config, **kwargs):
    np.random.seed(None)
    env = RecoEnv1()
    env.init_gym({**env_config, 'random_seed': np.random.randint(10000)})
    
    agent = agent_class(config, **kwargs)
    train_agent_offline(env, agent, COMPLEX_CONFIG['num_offline_users'])
    
    if hasattr(agent, 'update'):
        agent.update()
    
    ctr = evaluate_agent_online(env, agent, COMPLEX_CONFIG['num_online_users'])
    return ctr


def run_multiple_experiments(agent_class, config, env_config, n_runs=3, **kwargs):
    ctrs = []
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end='', flush=True)
        ctr = run_single_experiment(agent_class, config, env_config, **kwargs)
        ctrs.append(ctr)
        print(f" CTR={ctr:.4f}")
    return np.mean(ctrs), np.std(ctrs)


# ============================================
# 主实验
# ============================================

print("\n" + "=" * 70)
print("初始化复杂环境...")
print("=" * 70)

env = RecoEnv1()
env.init_gym(complex_env_config)
config = env.config

all_agents = {
    'Popularity': (PopularityAgent, {}),
    'ε-greedy(ε=0.1)': (EpsilonGreedyAgent, {'epsilon': 0.1}),
    'ε-greedy(ε=0.2)': (EpsilonGreedyAgent, {'epsilon': 0.2}),
    'UCB(c=1.0)': (UCBAgent, {'c': 1.0}),
    'UCB(c=2.0)': (UCBAgent, {'c': 2.0}),
    'Thompson Sampling': (ThompsonSamplingAgent, {}),
    'Policy Gradient': (PolicyGradientAgent, {'learning_rate': 0.5}),
    'DQN': (DQNAgent, {'learning_rate': 0.2}),
}

results = {}
print("\n" + "=" * 70)
print("运行复杂环境实验...")
print("=" * 70)

for name, (agent_class, kwargs) in all_agents.items():
    print(f"\n{name}:")
    mean_ctr, std_ctr = run_multiple_experiments(
        agent_class, config, complex_env_config, 
        n_runs=COMPLEX_CONFIG['num_runs'], 
        **kwargs
    )
    results[name] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"  平均: {mean_ctr:.4f} ± {std_ctr:.4f}")


# ============================================
# 可视化
# ============================================
print("\n生成图表...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1: 所有Agent对比
ax1 = axes[0]
names = list(results.keys())
means = [results[n]['mean'] for n in names]
stds = [results[n]['std'] for n in names]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']

bars = ax1.barh(names, means, xerr=stds, color=colors, capsize=3, alpha=0.8)
ax1.set_xlabel('CTR (点击率)')
ax1.set_title('复杂环境Agent性能对比 (100物品, K=20)')
ax1.set_xlim(0, max(means) * 1.3)

for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{mean:.4f}±{std:.4f}', ha='left', va='center', fontsize=9)

# 子图2: 探索策略对比
ax2 = axes[1]
explore_methods = ['ε-greedy\n(最优)', 'UCB\n(最优)', 'Thompson\nSampling']
explore_means = [
    max(results['ε-greedy(ε=0.1)']['mean'], results['ε-greedy(ε=0.2)']['mean']),
    max(results['UCB(c=1.0)']['mean'], results['UCB(c=2.0)']['mean']),
    results['Thompson Sampling']['mean']
]
explore_stds = [
    min(results['ε-greedy(ε=0.1)']['std'], results['ε-greedy(ε=0.2)']['std']),
    min(results['UCB(c=1.0)']['std'], results['UCB(c=2.0)']['std']),
    results['Thompson Sampling']['std']
]
colors_explore = ['#e74c3c', '#2ecc71', '#1abc9c']

bars2 = ax2.bar(explore_methods, explore_means, yerr=explore_stds, 
                color=colors_explore, capsize=5, alpha=0.8)
ax2.set_ylabel('CTR')
ax2.set_title('探索策略对比 (最优参数)')
ax2.set_ylim(0, max(explore_means) * 1.3)

for bar, mean, std in zip(bars2, explore_means, explore_stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results_v3_complex.png', 
            dpi=150, bbox_inches='tight')
print("图表已保存: results_v3_complex.png")


# ============================================
# 结果总结
# ============================================
print("\n" + "=" * 70)
print("实验结果总结")
print("=" * 70)

print("\nAgent性能排名（按CTR降序）:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
for rank, (name, result) in enumerate(sorted_results, 1):
    print(f"  {rank}. {name:20s}: {result['mean']:.4f} ± {result['std']:.4f}")

print("\n" + "=" * 70)
print("实验V3完成！结果已保存至 results_v3_complex.png")
print("=" * 70)
