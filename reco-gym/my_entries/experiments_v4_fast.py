"""
RecoGym V4 - 大规模环境快速版
"""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("RecoGym V4 - 大规模环境 (500物品)")
print("=" * 60)

NUM_PRODUCTS = 500
K = 10
OFFLINE = 200
ONLINE = 100

config_dict = {
    **env_1_args,
    'random_seed': 42,
    'num_products': NUM_PRODUCTS,
    'K': K,
    'sigma_omega': 0.3,
    'number_of_flips': 0,
}

print(f"\n配置: {NUM_PRODUCTS}物品, K={K}, 训练{OFFLINE}用户, 评估{ONLINE}用户")

class PopAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.views = np.zeros(config.num_products)
    def train(self, obs, act, r, done=False):
        if obs:
            for s in obs.sessions():
                self.views[s['v']] += 1
    def act(self, obs, r, done):
        a = np.argmax(self.views) if self.views.sum() > 0 else np.random.randint(self.config.num_products)
        return {'t': obs.context().time(), 'u': obs.context().user(), 'a': a, 'ps': 1.0, 'ps-a': np.zeros(self.config.num_products)}
    def reset(self):
        self.views = np.zeros(self.config.num_products)

class TSAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.n = config.num_products
        self.a = np.ones(self.n)
        self.b = np.ones(self.n)
    def train(self, obs, act, r, done=False):
        if act:
            if r > 0:
                self.a[act['a']] += 1
            else:
                self.b[act['a']] += 1
    def act(self, obs, r, done):
        samples = np.random.beta(self.a, self.b)
        a = np.argmax(samples)
        return {'t': obs.context().time(), 'u': obs.context().user(), 'a': a, 'ps': 1.0, 'ps-a': np.zeros(self.n)}
    def reset(self):
        self.a = np.ones(self.n)
        self.b = np.ones(self.n)

def run(env, agent, offline, online):
    uid = 0
    for _ in range(offline):
        env.reset(uid)
        uid += 1
        obs, _, done, _ = env.step(None)
        agent.train(obs, None, None, True)
        while not done:
            old = obs
            act, obs, r, done, _ = env.step_offline(old, 0, done)
            agent.train(old, act, r, False)
        act, obs, r, done, _ = env.step_offline(obs, 0, done)
        agent.train(obs, act, r, True)
    
    uid = 1000
    clicks, steps = 0, 0
    for _ in range(online):
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

env = RecoEnv1()
env.init_gym(config_dict)
config = env.config

results = {}
print("\n运行实验...")

for name, cls, kw in [('Popularity', PopAgent, {}), ('Thompson', TSAgent, {})]:
    print(f"  {name}...", end=' ', flush=True)
    np.random.seed(None)
    env2 = RecoEnv1()
    env2.init_gym({**config_dict, 'random_seed': np.random.randint(10000)})
    agent = cls(config, **kw)
    ctr = run(env2, agent, OFFLINE, ONLINE)
    results[name] = ctr
    print(f"CTR={ctr:.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(6, 4))
names = list(results.keys())
ctrs = [results[n] for n in names]
colors = ['#3498db', '#9b59b6']

bars = ax.bar(names, ctrs, color=colors, alpha=0.8)
ax.set_ylabel('CTR')
ax.set_title(f'大规模环境 ({NUM_PRODUCTS}物品)')
ax.set_ylim(0, max(ctrs) * 1.2)

for bar, c in zip(bars, ctrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{c:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results_v4_large.png', dpi=150)
print("\n图表已保存: results_v4_large.png")

print("\n" + "=" * 60)
print("结果:")
for n, c in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {n}: {c:.4f}")
print("=" * 60)
