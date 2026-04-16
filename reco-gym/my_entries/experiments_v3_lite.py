"""
RecoGym 强化学习推荐系统实验 V3 - 复杂环境版（精简）
================================
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

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("RecoGym V3 - 复杂环境版（精简）")
print("=" * 70)

# 复杂环境配置（精简版）
CONFIG = {
    'random_seed': 42,
    'num_products': 50,             # 50个物品
    'K': 10,                        # 潜在维度10
    'sigma_omega': 0.3,             # 用户兴趣变化
    'num_offline_users': 200,       # 训练用户
    'num_online_users': 100,        # 评估用户
    'num_runs': 2,                  # 2次实验
}

complex_env_config = {
    **env_1_args,
    'random_seed': CONFIG['random_seed'],
    'num_products': CONFIG['num_products'],
    'K': CONFIG['K'],
    'sigma_omega': CONFIG['sigma_omega'],
}

print(f"\n配置: {CONFIG['num_products']}物品, K={CONFIG['K']}, σ={CONFIG['sigma_omega']}")


# Agent定义
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
        return {'t': observation.context().time(), 'u': observation.context().user(), 
                'a': action, 'ps': 1.0, 'ps-a': np.zeros(self.config.num_products)}
    
    def reset(self):
        self.product_views = np.zeros(self.config.num_products)


class EpsilonGreedyAgent(Agent):
    def __init__(self, config, epsilon=0.1):
        super(EpsilonGreedyAgent, self).__init__(config)
        self.epsilon = epsilon
        self.num_products = config.num_products
        self.clicks = np.zeros(self.num_products)
        self.recs = np.zeros(self.num_products)
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            self.recs[action['a']] += 1
            self.clicks[action['a']] += reward
    
    def act(self, observation, reward, done):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_products)
        else:
            ctr = np.divide(self.clicks, self.recs, out=np.zeros(self.num_products), where=self.recs != 0)
            action = np.argmax(ctr)
        return {'t': observation.context().time(), 'u': observation.context().user(), 
                'a': action, 'ps': 1.0, 'ps-a': np.zeros(self.num_products)}
    
    def reset(self):
        self.clicks = np.zeros(self.num_products)
        self.recs = np.zeros(self.num_products)


class UCBAgent(Agent):
    def __init__(self, config, c=2.0):
        super(UCBAgent, self).__init__(config)
        self.c = c
        self.num_products = config.num_products
        self.Q = np.zeros(self.num_products)
        self.N = np.zeros(self.num_products)
        self.total = 0
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            a = action['a']
            self.N[a] += 1
            self.Q[a] += (reward - self.Q[a]) / self.N[a]
            self.total += 1
    
    def act(self, observation, reward, done):
        if self.total < self.num_products:
            action = self.total
        else:
            ucb = self.Q + self.c * np.sqrt(np.log(self.total + 1) / (self.N + 1e-8))
            action = np.argmax(ucb)
        return {'t': observation.context().time(), 'u': observation.context().user(), 
                'a': action, 'ps': 1.0, 'ps-a': np.zeros(self.num_products)}
    
    def reset(self):
        self.Q = np.zeros(self.num_products)
        self.N = np.zeros(self.num_products)
        self.total = 0


class ThompsonSamplingAgent(Agent):
    def __init__(self, config):
        super(ThompsonSamplingAgent, self).__init__(config)
        self.num_products = config.num_products
        self.alpha = np.ones(self.num_products)
        self.beta = np.ones(self.num_products)
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            if reward > 0:
                self.alpha[action['a']] += 1
            else:
                self.beta[action['a']] += 1
    
    def act(self, observation, reward, done):
        samples = np.random.beta(self.alpha, self.beta)
        action = np.argmax(samples)
        return {'t': observation.context().time(), 'u': observation.context().user(), 
                'a': action, 'ps': 1.0, 'ps-a': np.zeros(self.num_products)}
    
    def reset(self):
        self.alpha = np.ones(self.num_products)
        self.beta = np.ones(self.num_products)


# 训练和评估
def train_offline(env, agent, n_users):
    uid = 0
    for _ in range(n_users):
        env.reset(uid)
        uid += 1
        obs, _, done, _ = env.step(None)
        agent.train(obs, None, None, True)
        while not done:
            old_obs = obs
            act, obs, r, done, _ = env.step_offline(old_obs, 0, done)
            agent.train(old_obs, act, r, False)
        act, obs, r, done, _ = env.step_offline(obs, 0, done)
        agent.train(obs, act, r, True)


def evaluate_online(env, agent, n_users):
    uid = 1000
    clicks, steps = 0, 0
    for _ in range(n_users):
        env.reset(uid)
        uid += 1
        agent.reset()
        obs, _, done, _ = env.step(None)
        while not done:
            act = agent.act(obs, 0, done)
            obs, r, done, _ = env.step(act['a'])
            agent.train(obs, act, r, done)
            clicks += r
            steps += 1
    return clicks / steps if steps > 0 else 0


def run_experiment(agent_class, config, env_config, n_runs=2, **kwargs):
    ctrs = []
    for run in range(n_runs):
        np.random.seed(None)
        env = RecoEnv1()
        env.init_gym({**env_config, 'random_seed': np.random.randint(10000)})
        agent = agent_class(config, **kwargs)
        train_offline(env, agent, CONFIG['num_offline_users'])
        ctr = evaluate_online(env, agent, CONFIG['num_online_users'])
        ctrs.append(ctr)
        print(f"    Run {run+1}: CTR={ctr:.4f}")
    return np.mean(ctrs), np.std(ctrs)


# 主实验
env = RecoEnv1()
env.init_gym(complex_env_config)
config = env.config

agents = {
    'Popularity': (PopularityAgent, {}),
    'ε-greedy(ε=0.1)': (EpsilonGreedyAgent, {'epsilon': 0.1}),
    'ε-greedy(ε=0.2)': (EpsilonGreedyAgent, {'epsilon': 0.2}),
    'UCB(c=2.0)': (UCBAgent, {'c': 2.0}),
    'Thompson Sampling': (ThompsonSamplingAgent, {}),
}

results = {}
print("\n运行实验...")
for name, (cls, kw) in agents.items():
    print(f"\n{name}:")
    mean, std = run_experiment(cls, config, complex_env_config, n_runs=CONFIG['num_runs'], **kw)
    results[name] = {'mean': mean, 'std': std}
    print(f"  => {mean:.4f} ± {std:.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
means = [results[n]['mean'] for n in names]
stds = [results[n]['std'] for n in names]
colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

bars = ax.barh(names, means, xerr=stds, color=colors, capsize=3, alpha=0.8)
ax.set_xlabel('CTR')
ax.set_title(f'复杂环境Agent对比 ({CONFIG["num_products"]}物品, K={CONFIG["K"]})')
ax.set_xlim(0, max(means) * 1.3)

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{m:.4f}±{s:.4f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results_v3_complex.png', dpi=150)
print("\n图表已保存: results_v3_complex.png")

# 结果
print("\n" + "=" * 70)
print("结果排名:")
sorted_r = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
for i, (n, r) in enumerate(sorted_r, 1):
    print(f"  {i}. {n:20s}: {r['mean']:.4f} ± {r['std']:.4f}")

print("\n" + "=" * 70)
print("实验V3完成！")
print("=" * 70)
