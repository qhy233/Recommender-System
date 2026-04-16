"""
RecoGym 强化学习推荐系统实验 V2 - 简化版
================================
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("RecoGym 强化学习推荐系统实验 V2 - 简化版")
print("=" * 70)

# 简化配置
CONFIG = {
    'random_seed': 42,
    'num_products': 10,
    'num_offline_users': 100,
    'num_online_users': 50,
    'num_runs': 3,
}

config_dict = {
    **env_1_args,
    'random_seed': CONFIG['random_seed'],
    'num_products': CONFIG['num_products'],
}

# ============================================
# Agent定义
# ============================================

class PopularityAgent(Agent):
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
    train_agent_offline(env, agent, CONFIG['num_offline_users'])
    
    if hasattr(agent, 'update'):
        agent.update()
    
    ctr = evaluate_agent_online(env, agent, CONFIG['num_online_users'])
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

print(f"\n实验配置: {CONFIG['num_offline_users']} 离线用户, {CONFIG['num_online_users']} 在线用户, {CONFIG['num_runs']} 次运行")

env = RecoEnv1()
env.init_gym(config_dict)
config = env.config

all_agents = {
    'Popularity': (PopularityAgent, {}),
    'ε-greedy(ε=0.1)': (EpsilonGreedyAgent, {'epsilon': 0.1}),
    'ε-greedy(ε=0.3)': (EpsilonGreedyAgent, {'epsilon': 0.3}),
    'UCB(c=2.0)': (UCBAgent, {'c': 2.0}),
    'Thompson Sampling': (ThompsonSamplingAgent, {}),
}

results = {}
print("\n" + "=" * 70)
print("运行实验...")
print("=" * 70)

for name, (agent_class, kwargs) in all_agents.items():
    print(f"\n{name}:")
    mean_ctr, std_ctr = run_multiple_experiments(agent_class, config, config_dict, **kwargs)
    results[name] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"  平均: {mean_ctr:.4f} ± {std_ctr:.4f}")

# ============================================
# 可视化
# ============================================
print("\n生成图表...")

fig, ax = plt.subplots(figsize=(10, 6))

names = list(results.keys())
means = [results[n]['mean'] for n in names]
stds = [results[n]['std'] for n in names]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
ax.set_ylabel('CTR (点击率)')
ax.set_title('RecoGym Agent性能对比 (V2简化版)')
ax.set_ylim(0, max(means) * 1.3)

for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results_v2.png', 
            dpi=150, bbox_inches='tight')
print("图表已保存: results_v2.png")

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
print("实验V2完成！结果已保存至 results_v2.png")
print("=" * 70)
