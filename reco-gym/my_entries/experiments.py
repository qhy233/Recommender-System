"""
RecoGym 强化学习推荐系统实验
============================
使用本地RecoGym环境实现经典推荐算法：
1. 基于统计的推荐（协同过滤思想）
2. ε-greedy探索策略
3. 策略梯度推荐（REINFORCE）
4. DQN推荐（Value-based）
"""

import sys
import os

sys.path.insert(0, r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from recogym.envs import RecoEnv1, env_1_args
from recogym.agents import Agent

print("=" * 60)
print("RecoGym 强化学习推荐系统实验")
print("=" * 60)

# 配置环境参数
config_dict = {
    **env_1_args,
    'random_seed': 42,
    'num_products': 10,
}

# 创建环境
env = RecoEnv1()
env.init_gym(config_dict)
config = env.config

print(f"\n物品数量: {config.num_products}")
print(f"环境类型: RecoEnv1")
print("环境初始化完成！")

# 训练参数
NUM_OFFLINE_USERS = 100
NUM_ONLINE_USERS = 50

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
        self.buffer_size = 2000
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
# 训练和评估函数
# ============================================

def train_agent_offline(env, agent, num_users):
    """离线训练Agent"""
    unique_user_id = 0
    for _ in trange(num_users, desc='Offline Training'):
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
    
    for _ in trange(num_users, desc='Online Evaluation'):
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


# ============================================
# 实验1：基于统计的推荐Agent
# ============================================
print("\n" + "=" * 60)
print("实验1：基于统计的推荐Agent (Popularity)")
print("=" * 60)
print("""
核心思想：
- 统计每个物品被用户浏览的次数
- 推荐浏览次数最多的物品
- 对应理论：ItemCF中的物品流行度、协同过滤的群体智慧
""")

pop_agent = PopularityAgent(config)
train_agent_offline(env, pop_agent, NUM_OFFLINE_USERS)
print(f"物品浏览统计: {pop_agent.product_views.astype(int)}")

pop_ctr = evaluate_agent_online(env, pop_agent, NUM_ONLINE_USERS)
print(f"点击率(CTR): {pop_ctr:.4f}")

# ============================================
# 实验2：ε-greedy探索策略
# ============================================
print("\n" + "=" * 60)
print("实验2：ε-greedy探索策略")
print("=" * 60)
print("""
核心思想：
- 以概率ε随机探索（尝试新物品）
- 以概率1-ε利用当前最优（推荐点击率最高的物品）
- 对应理论：强化学习中的探索-利用困境
""")

epsilons = [0.0, 0.1, 0.2, 0.3]
results_eps = {}

for eps in epsilons:
    print(f"\n--- ε = {eps} ---")
    agent = EpsilonGreedyAgent(config, epsilon=eps)
    train_agent_offline(env, agent, NUM_OFFLINE_USERS)
    ctr = evaluate_agent_online(env, agent, NUM_ONLINE_USERS)
    results_eps[eps] = ctr
    print(f"ε={eps}: 点击率={ctr:.4f}")

eps_ctr = results_eps[0.1]

# ============================================
# 实验3：策略梯度推荐Agent (REINFORCE)
# ============================================
print("\n" + "=" * 60)
print("实验3：策略梯度推荐Agent (REINFORCE)")
print("=" * 60)
print("""
核心思想：
- 直接学习推荐策略π(a|s)
- 使用策略梯度更新参数
- 对应理论：论文03 YouTube REINFORCE

策略梯度公式：∇J(θ) = E[∇log π(a|s) * R]
""")

pg_agent = PolicyGradientAgent(config, learning_rate=0.5)
ctrs_pg = []

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1}/5 ---")
    train_agent_offline(env, pg_agent, NUM_OFFLINE_USERS)
    pg_agent.update()
    ctr = evaluate_agent_online(env, pg_agent, NUM_ONLINE_USERS)
    pg_agent.update()
    ctrs_pg.append(ctr)
    print(f"Epoch {epoch+1}: CTR = {ctr:.4f}")

pg_ctr = ctrs_pg[-1]
print(f"最终策略分布: {pg_agent.get_policy().round(3)}")

# ============================================
# 实验4：DQN推荐Agent
# ============================================
print("\n" + "=" * 60)
print("实验4：DQN推荐Agent")
print("=" * 60)
print("""
核心思想：
- 学习Q(s,a)价值函数
- 使用经验回放稳定训练
- 对应理论：论文02 DRN、Value-based方法

Q-learning更新：Q(s,a) = r + γ * max_a' Q(s',a')
""")

dqn_agent = DQNAgent(config, learning_rate=0.2)
ctrs_dqn = []

for epoch in range(5):
    print(f"\n--- Epoch {epoch+1}/5 ---")
    train_agent_offline(env, dqn_agent, NUM_OFFLINE_USERS)
    dqn_agent.update()
    ctr = evaluate_agent_online(env, dqn_agent, NUM_ONLINE_USERS)
    dqn_agent.update()
    ctrs_dqn.append(ctr)
    print(f"Epoch {epoch+1}: CTR = {ctr:.4f}, ε = {dqn_agent.epsilon:.4f}")

dqn_ctr = ctrs_dqn[-1]
print(f"最终Q值: {dqn_agent.Q.round(3)}")

# ============================================
# 实验5：性能对比分析
# ============================================
print("\n" + "=" * 60)
print("实验5：性能对比分析")
print("=" * 60)

results = {
    'Popularity': pop_ctr,
    'ε-greedy(0.1)': eps_ctr,
    'Policy Gradient': pg_ctr,
    'DQN': dqn_ctr
}

print("\n各Agent点击率(CTR)对比:")
print("-" * 50)
for name, ctr in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(ctr * 200)
    print(f"{name:20s}: {ctr:.4f} {bar}")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1：各算法CTR对比
ax1 = axes[0]
names = list(results.keys())
ctrs = list(results.values())
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax1.bar(names, ctrs, color=colors)
ax1.set_ylabel('CTR')
ax1.set_title('Algorithm CTR Comparison')
ax1.set_ylim(0, max(ctrs) * 1.3 if max(ctrs) > 0 else 0.1)
for bar, ctr in zip(bars, ctrs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{ctr:.4f}', ha='center', va='bottom', fontsize=10)

# 图2：ε-greedy不同ε值对比
ax2 = axes[1]
eps_names = [f'ε={e}' for e in epsilons]
eps_ctrs = [results_eps[e] for e in epsilons]
bars2 = ax2.bar(eps_names, eps_ctrs, color='#9b59b6')
ax2.set_ylabel('CTR')
ax2.set_title('Epsilon-Greedy Exploration Rate')
ax2.set_ylim(0, max(eps_ctrs) * 1.3 if max(eps_ctrs) > 0 else 0.1)
for bar, ctr in zip(bars2, eps_ctrs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{ctr:.4f}', ha='center', va='bottom', fontsize=10)

# 图3：训练曲线
ax3 = axes[2]
ax3.plot(range(1, 6), ctrs_pg, 'g-o', label='Policy Gradient', linewidth=2, markersize=8)
ax3.plot(range(1, 6), ctrs_dqn, 'r-s', label='DQN', linewidth=2, markersize=8)
ax3.axhline(y=pop_ctr, color='b', linestyle='--', label='Popularity baseline')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('CTR')
ax3.set_title('Training Progress')
ax3.legend()
ax3.set_ylim(0, max(max(ctrs_pg), max(ctrs_dqn), pop_ctr) * 1.2 if max(max(ctrs_pg), max(ctrs_dqn), pop_ctr) > 0 else 0.1)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\LENOVO\Desktop\vs\fun-rec\reco-gym\my_entries\results.png', dpi=150, bbox_inches='tight')
print("\n结果图表已保存至: results.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
