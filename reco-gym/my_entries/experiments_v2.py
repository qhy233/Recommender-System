"""
RecoGym 强化学习推荐系统实验 V2
================================
扩展版本，新增内容：
1. 加大样本量 + 多次实验统计显著性
2. UCB Agent (Upper Confidence Bound)
3. Thompson Sampling Agent (贝叶斯方法)
4. 消融实验（探索率、学习率、样本量影响）
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from typing import Dict, List, Tuple
import time

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("RecoGym 强化学习推荐系统实验 V2 - 扩展版")
print("=" * 70)

# ============================================
# 实验配置
# ============================================
CONFIG = {
    'random_seed': 42,
    'num_products': 10,
    'num_offline_users': 200,      # 简化：减少样本量
    'num_online_users': 100,       # 简化：减少样本量
    'num_runs': 3,                 # 简化：减少实验次数
}

config_dict = {
    **env_1_args,
    'random_seed': CONFIG['random_seed'],
    'num_products': CONFIG['num_products'],
}

env = RecoEnv1()
env.init_gym(config_dict)
config = env.config

print(f"\n实验配置:")
print(f"  物品数量: {CONFIG['num_products']}")
print(f"  离线训练用户数: {CONFIG['num_offline_users']}")
print(f"  在线评估用户数: {CONFIG['num_online_users']}")
print(f"  重复实验次数: {CONFIG['num_runs']}")


# ============================================
# 基础Agent定义（保留原有）
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
    
    def reset(self):
        pass


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
            if np.max(click_rates) == 0:
                action = np.random.randint(self.config.num_products)
            else:
                action = np.argmax(click_rates)
        
        ps = self.epsilon / self.config.num_products if np.random.random() < self.epsilon else 1 - self.epsilon + self.epsilon / self.config.num_products
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': ps,
            'ps-a': np.zeros(self.config.num_products)
        }
    
    def reset(self):
        pass


class PolicyGradientAgent(Agent):
    """策略梯度推荐Agent (REINFORCE)"""
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
        pass


class DQNAgent(Agent):
    """DQN推荐Agent"""
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
        ps = self.epsilon / self.num_products if np.random.random() < self.epsilon else 1 - self.epsilon + self.epsilon / self.num_products
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': ps,
            'ps-a': np.zeros(self.config.num_products)
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
        if len(self.replay_buffer) < 32:
            return
        batch_size = min(64, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        for idx in indices:
            exp = self.replay_buffer[idx]
            action = exp['action']
            reward = exp['reward']
            self.Q[action] += self.lr * (reward - self.Q[action])
        self.epsilon = max(0.05, self.epsilon * 0.995)
    
    def reset(self):
        pass


# ============================================
# 新增Agent: UCB (Upper Confidence Bound)
# ============================================

class UCBAgent(Agent):
    """
    UCB探索策略Agent
    
    核心思想：
    - Upper Confidence Bound，基于置信上界选择动作
    - 平衡探索（不确定性高的物品）和利用（平均奖励高的物品）
    - 公式：UCB(a) = Q(a) + c * sqrt(ln(N) / n(a))
      - Q(a): 物品a的平均奖励
      - N: 总选择次数
      - n(a): 物品a被选择次数
      - c: 探索参数
    
    理论保证：
    - UCB是遗憾界最优的算法
    - 遗憾 = 最优策略累积奖励 - 实际累积奖励
    """
    def __init__(self, config, c=2.0):
        super(UCBAgent, self).__init__(config)
        self.c = c  # 探索参数
        self.num_products = config.num_products
        self.Q = np.zeros(self.num_products)           # 每个物品的平均奖励
        self.N = np.zeros(self.num_products)           # 每个物品被选择次数
        self.total_count = 0                           # 总选择次数
    
    def train(self, observation, action, reward, done=False):
        if action is not None:
            a = action['a']
            self.N[a] += 1
            self.total_count += 1
            self.Q[a] += (reward - self.Q[a]) / self.N[a]
    
    def act(self, observation, reward, done):
        if self.total_count < self.num_products:
            action = self.total_count
        else:
            confidence = self.c * np.sqrt(np.log(self.total_count + 1) / (self.N + 1e-8))
            ucb_values = self.Q + confidence
            action = np.argmax(ucb_values)
        
        ps = 1.0 / self.num_products
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': ps,
            'ps-a': np.ones(self.num_products) * ps
        }
    
    def reset(self):
        pass


# ============================================
# 新增Agent: Thompson Sampling
# ============================================

class ThompsonSamplingAgent(Agent):
    """
    Thompson Sampling探索策略Agent
    
    核心思想：
    - 贝叶斯方法，对每个物品维护奖励概率的后验分布
    - 每次从后验分布采样，选择采样值最大的物品
    - 对于伯努利奖励（点击/不点击），使用Beta分布作为共轭先验
    
    公式：
    - 先验：Beta(α, β)，初始α=β=1（均匀分布）
    - 观察到点击：α += 1
    - 观察到不点击：β += 1
    - 后验：Beta(α, β)
    - 采样：从Beta(α, β)采样，选择采样值最大的物品
    
    优势：
    - 自然平衡探索与利用
    - 计算简单，效果通常优于UCB
    - 可以自然地融入先验知识
    """
    def __init__(self, config):
        super(ThompsonSamplingAgent, self).__init__(config)
        self.num_products = config.num_products
        self.alpha = np.ones(self.num_products)   # Beta分布参数α（成功次数+1）
        self.beta = np.ones(self.num_products)    # Beta分布参数β（失败次数+1）
    
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
        
        ps = samples[action] / np.sum(samples)
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
            'a': action,
            'ps': ps,
            'ps-a': samples / np.sum(samples)
        }
    
    def reset(self):
        pass
    
    def get_estimated_rates(self):
        """获取每个物品的估计点击率"""
        return self.alpha / (self.alpha + self.beta)


# ============================================
# 训练和评估函数
# ============================================

def train_agent_offline(env, agent, num_users):
    """离线训练Agent"""
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
    """在线评估Agent"""
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


def run_multiple_experiments(agent_class, config, env_config, n_runs=5, 
                             num_offline=500, num_online=200, **kwargs):
    """
    多次实验取平均，计算均值和标准差
    
    参数:
        agent_class: Agent类
        config: 环境配置
        env_config: 环境参数字典
        n_runs: 实验次数
        num_offline: 离线训练用户数
        num_online: 在线评估用户数
        **kwargs: Agent初始化参数
    
    返回:
        (mean_ctr, std_ctr): 平均CTR和标准差
    """
    ctrs = []
    
    for run in range(n_runs):
        np.random.seed(42 + run)
        env = RecoEnv1()
        env.init_gym({**env_config, 'random_seed': 42 + run})
        
        agent = agent_class(config, **kwargs)
        train_agent_offline(env, agent, num_offline)
        
        if hasattr(agent, 'update'):
            agent.update()
        
        ctr = evaluate_agent_online(env, agent, num_online)
        ctrs.append(ctr)
    
    return np.mean(ctrs), np.std(ctrs)


# ============================================
# 实验1：所有Agent对比（加大样本量）
# ============================================
print("\n" + "=" * 70)
print("实验1：所有Agent对比（加大样本量 + 多次实验）")
print("=" * 70)

all_agents = {
    'Popularity': (PopularityAgent, {}),
    'ε-greedy(ε=0.1)': (EpsilonGreedyAgent, {'epsilon': 0.1}),
    'UCB(c=2.0)': (UCBAgent, {'c': 2.0}),
    'Thompson Sampling': (ThompsonSamplingAgent, {}),
    'Policy Gradient': (PolicyGradientAgent, {'learning_rate': 0.5}),
    'DQN': (DQNAgent, {'learning_rate': 0.2}),
}

results_exp1 = {}
print(f"\n运行 {CONFIG['num_runs']} 次实验，每次 {CONFIG['num_offline_users']} 离线用户 + {CONFIG['num_online_users']} 在线用户")
print("-" * 70)

for name, (agent_class, kwargs) in tqdm(all_agents.items(), desc="Agent对比"):
    mean_ctr, std_ctr = run_multiple_experiments(
        agent_class, config, config_dict,
        n_runs=CONFIG['num_runs'],
        num_offline=CONFIG['num_offline_users'],
        num_online=CONFIG['num_online_users'],
        **kwargs
    )
    results_exp1[name] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"{name:25s}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")


# ============================================
# 实验2：探索策略消融实验
# ============================================
print("\n" + "=" * 70)
print("实验2：探索策略消融实验")
print("=" * 70)

# 2.1 ε-greedy不同ε值
print("\n--- 2.1 ε-greedy不同探索率 ---")
epsilons = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
results_eps = {}

for eps in tqdm(epsilons, desc="ε值实验"):
    mean_ctr, std_ctr = run_multiple_experiments(
        EpsilonGreedyAgent, config, config_dict,
        n_runs=CONFIG['num_runs'],
        num_offline=CONFIG['num_offline_users'],
        num_online=CONFIG['num_online_users'],
        epsilon=eps
    )
    results_eps[eps] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"ε = {eps:.2f}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")

# 2.2 UCB不同c值
print("\n--- 2.2 UCB不同探索参数c ---")
c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
results_ucb = {}

for c in tqdm(c_values, desc="UCB c值实验"):
    mean_ctr, std_ctr = run_multiple_experiments(
        UCBAgent, config, config_dict,
        n_runs=CONFIG['num_runs'],
        num_offline=CONFIG['num_offline_users'],
        num_online=CONFIG['num_online_users'],
        c=c
    )
    results_ucb[c] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"c = {c:.1f}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")


# ============================================
# 实验3：学习率消融实验
# ============================================
print("\n" + "=" * 70)
print("实验3：学习率消融实验")
print("=" * 70)

# 3.1 Policy Gradient学习率
print("\n--- 3.1 Policy Gradient不同学习率 ---")
lr_pg = [0.01, 0.1, 0.5, 1.0, 2.0]
results_lr_pg = {}

for lr in tqdm(lr_pg, desc="PG学习率"):
    mean_ctr, std_ctr = run_multiple_experiments(
        PolicyGradientAgent, config, config_dict,
        n_runs=CONFIG['num_runs'],
        num_offline=CONFIG['num_offline_users'],
        num_online=CONFIG['num_online_users'],
        learning_rate=lr
    )
    results_lr_pg[lr] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"lr = {lr:.2f}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")

# 3.2 DQN学习率
print("\n--- 3.2 DQN不同学习率 ---")
lr_dqn = [0.01, 0.1, 0.2, 0.5, 1.0]
results_lr_dqn = {}

for lr in tqdm(lr_dqn, desc="DQN学习率"):
    mean_ctr, std_ctr = run_multiple_experiments(
        DQNAgent, config, config_dict,
        n_runs=CONFIG['num_runs'],
        num_offline=CONFIG['num_offline_users'],
        num_online=CONFIG['num_online_users'],
        learning_rate=lr
    )
    results_lr_dqn[lr] = {'mean': mean_ctr, 'std': std_ctr}
    print(f"lr = {lr:.2f}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")


# ============================================
# 实验4：样本量影响实验
# ============================================
print("\n" + "=" * 70)
print("实验4：样本量影响实验")
print("=" * 70)

offline_users_list = [100, 200, 500, 1000, 2000]
sample_agents = ['Popularity', 'UCB(c=2.0)', 'Thompson Sampling']
results_sample = {agent: {} for agent in sample_agents}

for num_offline in tqdm(offline_users_list, desc="样本量实验"):
    print(f"\n--- 离线用户数 = {num_offline} ---")
    
    for name in sample_agents:
        agent_class, kwargs = all_agents[name]
        mean_ctr, std_ctr = run_multiple_experiments(
            agent_class, config, config_dict,
            n_runs=CONFIG['num_runs'],
            num_offline=num_offline,
            num_online=CONFIG['num_online_users'],
            **kwargs
        )
        results_sample[name][num_offline] = {'mean': mean_ctr, 'std': std_ctr}
        print(f"{name:25s}: CTR = {mean_ctr:.4f} ± {std_ctr:.4f}")


# ============================================
# 可视化
# ============================================
print("\n" + "=" * 70)
print("生成可视化图表...")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))

# 图1：所有Agent对比
ax1 = fig.add_subplot(2, 3, 1)
names = list(results_exp1.keys())
means = [results_exp1[n]['mean'] for n in names]
stds = [results_exp1[n]['std'] for n in names]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bars = ax1.barh(names, means, xerr=stds, color=colors, capsize=3, alpha=0.8)
ax1.set_xlabel('CTR')
ax1.set_title('所有Agent对比 (均值±标准差)')
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{mean:.4f}±{std:.4f}', va='center', fontsize=9)

# 图2：ε-greedy探索率
ax2 = fig.add_subplot(2, 3, 2)
eps_vals = list(results_eps.keys())
eps_means = [results_eps[e]['mean'] for e in eps_vals]
eps_stds = [results_eps[e]['std'] for e in eps_vals]
ax2.errorbar(eps_vals, eps_means, yerr=eps_stds, fmt='o-', capsize=5, 
             color='#e74c3c', linewidth=2, markersize=8)
ax2.set_xlabel('探索率 ε')
ax2.set_ylabel('CTR')
ax2.set_title('ε-greedy: 探索率影响')
ax2.grid(True, alpha=0.3)
best_eps = eps_vals[np.argmax(eps_means)]
ax2.axvline(x=best_eps, color='green', linestyle='--', alpha=0.7, label=f'最优ε={best_eps}')
ax2.legend()

# 图3：UCB探索参数
ax3 = fig.add_subplot(2, 3, 3)
c_vals = list(results_ucb.keys())
c_means = [results_ucb[c]['mean'] for c in c_vals]
c_stds = [results_ucb[c]['std'] for c in c_vals]
ax3.errorbar(c_vals, c_means, yerr=c_stds, fmt='s-', capsize=5,
             color='#2ecc71', linewidth=2, markersize=8)
ax3.set_xlabel('探索参数 c')
ax3.set_ylabel('CTR')
ax3.set_title('UCB: 探索参数影响')
ax3.grid(True, alpha=0.3)
best_c = c_vals[np.argmax(c_means)]
ax3.axvline(x=best_c, color='red', linestyle='--', alpha=0.7, label=f'最优c={best_c}')
ax3.legend()

# 图4：学习率对比
ax4 = fig.add_subplot(2, 3, 4)
pg_lr_vals = list(results_lr_pg.keys())
pg_lr_means = [results_lr_pg[lr]['mean'] for lr in pg_lr_vals]
dqn_lr_vals = list(results_lr_dqn.keys())
dqn_lr_means = [results_lr_dqn[lr]['mean'] for lr in dqn_lr_vals]
ax4.plot(pg_lr_vals, pg_lr_means, 'g-o', label='Policy Gradient', linewidth=2, markersize=8)
ax4.plot(dqn_lr_vals, dqn_lr_means, 'r-s', label='DQN', linewidth=2, markersize=8)
ax4.set_xlabel('学习率')
ax4.set_ylabel('CTR')
ax4.set_title('学习率影响对比')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 图5：样本量影响
ax5 = fig.add_subplot(2, 3, 5)
for agent_name in sample_agents:
    x_vals = list(results_sample[agent_name].keys())
    y_vals = [results_sample[agent_name][x]['mean'] for x in x_vals]
    ax5.plot(x_vals, y_vals, 'o-', label=agent_name, linewidth=2, markersize=8)
ax5.set_xlabel('离线训练用户数')
ax5.set_ylabel('CTR')
ax5.set_title('样本量影响')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')

# 图6：探索策略对比总结
ax6 = fig.add_subplot(2, 3, 6)
explore_methods = ['ε-greedy(最优)', 'UCB(最优)', 'Thompson Sampling']
explore_means = [
    max(eps_means),
    max(c_means),
    results_exp1['Thompson Sampling']['mean']
]
explore_stds = [
    results_eps[best_eps]['std'],
    results_ucb[best_c]['std'],
    results_exp1['Thompson Sampling']['std']
]
colors_explore = ['#e74c3c', '#2ecc71', '#f39c12']
bars6 = ax6.bar(explore_methods, explore_means, yerr=explore_stds, 
                color=colors_explore, capsize=5, alpha=0.8)
ax6.set_ylabel('CTR')
ax6.set_title('探索策略对比 (最优参数)')
ax6.set_ylim(0, max(explore_means) * 1.3)
for bar, mean, std in zip(bars6, explore_means, explore_stds):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results_v2.png', 
            dpi=150, bbox_inches='tight')
print("结果图表已保存至: results_v2.png")


# ============================================
# 实验总结
# ============================================
print("\n" + "=" * 70)
print("实验总结")
print("=" * 70)

print("\nAgent性能排名（按CTR降序）：")
sorted_results = sorted(results_exp1.items(), key=lambda x: x[1]['mean'], reverse=True)
for rank, (name, result) in enumerate(sorted_results, 1):
    print(f"  {rank}. {name:25s}: {result['mean']:.4f} ± {result['std']:.4f}")

print("\n" + "=" * 70)
print("实验V2完成！结果已保存至 results_v2.png")
print("=" * 70)
