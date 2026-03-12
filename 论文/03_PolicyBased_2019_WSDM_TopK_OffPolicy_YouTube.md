# 03_PolicyBased_2019_WSDM_TopK_OffPolicy_YouTube

**论文信息**
- 标题：Top-K Off-Policy Correction for a REINFORCE Recommender System
- 作者：Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi
- 单位：Google, Inc.
- 发表：WSDM 2019

---

## 一、核心痛点与动机 (The "Why")

### 1.1 工业推荐系统的独特挑战

> "Industrial recommender systems deal with extremely large action spaces – many millions of items to recommend. Moreover, they need to serve billions of users, who are unique at any point in time, making a complex user state space."

**翻译**：工业推荐系统处理极大的动作空间——数百万个待推荐物品。此外，它们需要服务数十亿用户，这些用户在任何时刻都是独特的，构成了复杂的用户状态空间。

**核心挑战**：
| 挑战 | 描述 | 影响 |
|------|------|------|
| 极大动作空间 | 数百万个待推荐视频 | 传统RL算法难以扩展 |
| 极大状态空间 | 数十亿用户，每个用户状态独特 | 状态表示复杂 |
| 数据稀疏性 | Netflix Prize数据集仅0.1%密集 | 训练数据不足 |
| 非实时交互 | 无法进行在线策略更新和即时轨迹生成 | 需要off-policy学习 |
| 多策略数据 | 数据来自多个历史策略的混合 | 数据分布偏差 |

### 1.2 经典RL与推荐系统的差异

> "Unlike classical reinforcement learning, our learner does not have real-time interactive control of the recommender due to learning and infrastructure constraints. In other words, we cannot perform online updates to the policy and generate trajectories according to the updated policy immediately."

**翻译**：与经典强化学习不同，由于学习和基础设施限制，我们的学习器无法对推荐系统进行实时交互控制。换句话说，我们无法执行在线策略更新并立即根据更新后的策略生成轨迹。

**关键差异**：
- **经典RL**：可以实时与环境交互，在线更新策略
- **推荐系统RL**：
  - 只能从历史策略收集的日志反馈中学习
  - 无法实时探索新的状态-动作空间
  - 需要处理多个历史策略的数据偏差

### 1.3 Top-K推荐的特殊性

> "Finally, most of the research in RL focuses on producing a policy that chooses a single item. Real-world recommenders, on the other hand, typically offer the user multiple recommendations at a time."

**翻译**：最后，大多数RL研究专注于生成选择单个物品的策略。而现实世界的推荐系统通常一次向用户提供多个推荐。

**核心问题**：
- 传统RL：选择单个物品
- 实际推荐：一次推荐Top-K个物品
- 需要专门的Top-K off-policy correction

---

## 二、核心创新点

### 2.1 REINFORCE推荐器：扩展到极大动作空间

> "We scale a REINFORCE policy-gradient-based approach to learn a neural recommendation policy in a extremely large action space."

**翻译**：我们将REINFORCE策略梯度方法扩展到极大动作空间中学习神经推荐策略。

**为什么选择REINFORCE而非DQN**：
> "Although value-based methods present many advantages such as seamless off-policy learning, they are known to be prone to instability with function approximation. Often, extensive hyper-parameter tuning is required to achieve stable behavior for these approaches. Policy-based approaches on the other hand, remain rather stable w.r.t. function approximations given a sufficiently small learning rate."

**翻译**：虽然基于值的方法具有许多优势，如无缝的off-policy学习，但它们在函数近似下容易出现不稳定。通常需要大量的超参数调整才能实现稳定行为。另一方面，基于策略的方法在给定足够小的学习率的情况下，对函数近似保持相当稳定。

**优势对比**：
| 方法 | 优点 | 缺点 |
|------|------|------|
| Value-based (DQN) | 无缝off-policy学习 | 函数近似不稳定，需大量调参 |
| Policy-based (REINFORCE) | 函数近似稳定 | 需要off-policy correction |

### 2.2 Off-Policy候选生成：处理数据偏差

> "We apply off-policy correction to learn from logged feedback, collected from an ensemble of prior model policies. We incorporate a learned neural model of the behavior policies to correct data biases."

**翻译**：我们应用off-policy correction从日志反馈中学习，这些反馈是从先前模型策略的集成中收集的。我们结合学习到的行为策略神经模型来纠正数据偏差。

**核心问题**：
- 数据来自历史策略 $\beta$，而非当前策略 $\pi_\theta$
- 朴素策略梯度估计器不再无偏
- 需要重要性采样权重进行修正

**重要性采样权重**：
$$\omega(s_t, a_t) = \frac{d^\pi_t(s_t)}{d^\beta_t(s_t)} \times \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} \times \prod_{t'=t+1}^{|\tau|} \frac{\pi_\theta(a_{t'}|s_{t'})}{\beta_\theta(a_{t'}|s_{t'})}$$

**一阶近似**：
> "To reduce the variance of each gradient term, we take the first-order approximation and ignore the state visitation differences under the two policies as the importance weights of future trajectories, which yields a slightly biased estimator of the policy gradient with lower variance."

**翻译**：为了减少每个梯度项的方差，我们采用一阶近似，忽略两个策略下的状态访问差异作为未来轨迹的重要性权重，这产生了一个方差较低的策略梯度略微有偏估计器。

$$\nabla_\theta J(\pi_\theta) \approx \sum_{s_t \sim d^\beta_t(\cdot), a_t \sim \beta(\cdot|s_t)} \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} R_t \nabla_\theta \log \pi_\theta(\tau)$$

### 2.3 Top-K Off-Policy Correction：核心创新

> "We offer a novel top-K off-policy correction to account for the fact that our recommender outputs multiple items at a time."

**翻译**：我们提出了一种新颖的Top-K off-policy correction，以考虑到我们的推荐器一次输出多个物品的事实。

**核心洞察**：
- 标准off-policy correction优化Top-1推荐
- Top-K推荐需要不同的correction
- 需要考虑K个物品的联合分布

**数学推导**：
对于Top-K推荐，策略选择K个物品的集合：
$$\pi_\theta(a_1, a_2, ..., a_K | s)$$

Top-K off-policy correction：
$$\omega_{top-K}(s_t, a_t) = \frac{\pi_\theta(a_t \in \text{Top-K}|s_t)}{\beta(a_t \in \text{Top-K}|s_t)}$$

**实验验证**：
> "We find that while the standard off-policy correction results in a policy that is optimal for top-1 recommendation, this top-K off-policy correction leads to significant better top-K recommendations in both simulations and live experiments."

**翻译**：我们发现，虽然标准off-policy correction产生的策略对Top-1推荐是最优的，但这种Top-K off-policy correction在模拟和在线实验中都导致了显著更好的Top-K推荐。

### 2.4 探索的价值

> "We demonstrate the efficacy of our approaches through a series of simulations and multiple live experiments on YouTube."

**翻译**：我们通过一系列模拟和YouTube上的多次在线实验证明了我们方法的有效性。

**探索策略**：
- 在策略中引入随机性
- 发现新的有吸引力的物品
- 避免信息茧房

---

## 三、模型架构

### 3.1 MDP建模

> "We translate this setup into a Markov Decision Process (MDP) $(S, A, P, R, \rho_0, \gamma)$"

**MDP要素**：
| 要素 | 定义 | 推荐系统含义 |
|------|------|--------------|
| $S$ | 连续状态空间 | 用户状态（兴趣、历史行为） |
| $A$ | 离散动作空间 | 待推荐的视频集合 |
| $P$ | 状态转移概率 | 用户状态演化 |
| $R$ | 奖励函数 | 点击、观看时长等用户满意度指标 |
| $\rho_0$ | 初始状态分布 | 用户初始状态 |
| $\gamma$ | 折扣因子 | 未来奖励的折扣 |

**目标**：
$$\max_\pi J(\pi) = \mathbb{E}_{\tau \sim \pi}[R(\tau)]$$

其中 $R(\tau) = \sum_{t=0}^{|\tau|} r(s_t, a_t)$

### 3.2 REINFORCE梯度

**策略梯度公式**：
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s_t \sim d^\pi_t(\cdot), a_t \sim \pi(\cdot|s_t)} \left[ R_t(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$

其中 $R_t(s_t, a_t) = \sum_{t'=t}^{|\tau|} \gamma^{t'-t} r(s_{t'}, a_{t'})$ 是折扣未来奖励。

### 3.3 策略参数化

#### 3.3.1 状态表示

> "We model our belief on the user state at each time t, which capture both evolving user interests using a n-dimensional vector, that is, $s_t \in \mathbb{R}^n$."

**翻译**：我们对每个时间t的用户状态进行建模，使用n维向量捕捉演化的用户兴趣，即 $s_t \in \mathbb{R}^n$。

#### 3.3.2 动作嵌入

> "The action taken at each time t along the trajectory is embedded using an m-dimensional vector $u_{a_t} \in \mathbb{R}^m$."

**翻译**：轨迹上每个时间t采取的动作使用m维向量 $u_{a_t} \in \mathbb{R}^m$ 嵌入。

#### 3.3.3 状态转移（RNN）

> "We model the state transition P: S×A×S with a recurrent neural network: $s_{t+1} = f(s_t, u_{a_t})$."

**翻译**：我们使用循环神经网络建模状态转移：$s_{t+1} = f(s_t, u_{a_t})$。

**CFN (Chaos Free RNN) 单元**：
> "We experimented with a variety of popular RNN cells such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), and ended up using a simplified cell called Chaos Free RNN (CFN) due to its stability and computational efficiency."

**翻译**：我们尝试了多种流行的RNN单元，如LSTM和GRU，最终由于其稳定性和计算效率使用了称为Chaos Free RNN (CFN)的简化单元。

**状态更新公式**：
$$s_{t+1} = z_t \odot \tanh(s_t) + i_t \odot \tanh(W_a u_{a_t})$$

其中：
- $z_t = \sigma(U_z s_t + W_z u_{a_t} + b_z)$：更新门
- $i_t = \sigma(U_i s_t + W_i u_{a_t} + b_i)$：输入门

#### 3.3.4 策略输出（Softmax）

> "Conditioning on a user state s, the policy $\pi_\theta(a|s)$ is then modeled with a simple softmax."

**翻译**：基于用户状态s，策略 $\pi_\theta(a|s)$ 使用简单的softmax建模。

**策略公式**：
$$\pi_\theta(a|s) = \frac{\exp(s^\top v_a / T)}{\sum_{a' \in A} \exp(s^\top v_{a'} / T)}$$

其中：
- $v_a \in \mathbb{R}^n$：动作a的另一个嵌入
- $T$：温度参数（通常设为1）

**计算优化**：
> "The normalization term in the softmax requires going over all the possible actions, which is in the order of millions in our setting. To speed up the computation, we perform sampled softmax during training. At serving time, we used an efficient nearest neighbor search algorithm to retrieve top actions and approximate the softmax probability using these actions only."

**翻译**：softmax中的归一化项需要遍历所有可能的动作，在我们的设置中数量级为数百万。为了加速计算，我们在训练期间执行采样softmax。在服务时，我们使用高效的最近邻搜索算法检索top动作，并仅使用这些动作近似softmax概率。

---

## 四、Off-Policy Correction详解

### 4.1 问题定义

**数据收集机制**：
- 数据来自历史策略 $\beta$（或多个策略的混合）
- 当前策略 $\pi_\theta$ 与 $\beta$ 分布不同
- 朴素策略梯度估计器有偏

### 4.2 重要性采样修正

**完整重要性权重**：
$$\omega(s_t, a_t) = \frac{d^\pi_t(s_t)}{d^\beta_t(s_t)} \times \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} \times \prod_{t'=t+1}^{|\tau|} \frac{\pi_\theta(a_{t'}|s_{t'})}{\beta_\theta(a_{t'}|s_{t'})}$$

**问题**：
- 方差巨大
- 当 $\pi_\theta$ 和 $\beta$ 差异大时，重要性权重可能极低或极高

### 4.3 一阶近似

**简化公式**：
$$\nabla_\theta J(\pi_\theta) \approx \sum_{s_t \sim d^\beta_t(\cdot), a_t \sim \beta(\cdot|s_t)} \frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)} R_t \nabla_\theta \log \pi_\theta(\tau)$$

**理论保证**：
> "Achiam et al. [1] prove that the impact of this first-order approximation on the total reward of the learned policy is bounded in magnitude by $O(\mathbb{E}_{s \sim d^\beta}[D_{TV}(\pi||\beta)[s]])$ where $D_{TV}$ is the total variation between $\pi(\cdot|s)$ and $\beta(\cdot|s)$."

**翻译**：Achiam等人证明，这种一阶近似对学习策略的总奖励的影响在幅度上受 $O(\mathbb{E}_{s \sim d^\beta}[D_{TV}(\pi||\beta)[s]])$ 限制，其中 $D_{TV}$ 是 $\pi(\cdot|s)$ 和 $\beta(\cdot|s)$ 之间的总变差。

**权衡**：
- 牺牲少量偏差
- 大幅降低方差
- 更适合实际应用

### 4.4 行为策略建模

> "We incorporate a learned neural model of the behavior policies to correct data biases."

**翻译**：我们结合学习到的行为策略神经模型来纠正数据偏差。

**实现方式**：
- 使用神经网络建模历史策略 $\beta$
- 从日志数据中学习 $\beta(a|s)$
- 用于计算重要性权重

---

## 五、Top-K Off-Policy Correction

### 5.1 动机

**核心问题**：
- 标准off-policy correction优化Top-1推荐
- 实际推荐系统输出Top-K物品
- 需要专门的Top-K correction

### 5.2 数学推导

**Top-K策略**：
$$\pi_\theta(a_1, a_2, ..., a_K | s)$$

**Top-K重要性权重**：
$$\omega_{top-K}(s_t, a_t) = \frac{\pi_\theta(a_t \in \text{Top-K}|s_t)}{\beta(a_t \in \text{Top-K}|s_t)}$$

### 5.3 实验验证

**关键发现**：
> "We find that while the standard off-policy correction results in a policy that is optimal for top-1 recommendation, this top-K off-policy correction leads to significant better top-K recommendations in both simulations and live experiments."

**翻译**：我们发现，虽然标准off-policy correction产生的策略对Top-1推荐是最优的，但这种Top-K off-policy correction在模拟和在线实验中都导致了显著更好的Top-K推荐。

**性能对比**：
| 方法 | Top-1性能 | Top-K性能 |
|------|-----------|-----------|
| 标准off-policy correction | 最优 | 次优 |
| Top-K off-policy correction | 次优 | 显著更好 |

---

## 六、实验结果

### 6.1 实验设置

**数据来源**：
- YouTube生产环境的真实用户数据
- 数百万视频候选
- 数十亿用户

**评估方式**：
- 离线模拟
- 在线A/B测试

### 6.2 关键指标

| 指标 | 说明 |
|------|------|
| 点击率 (CTR) | 用户点击推荐视频的比例 |
| 观看时长 | 用户观看推荐视频的总时长 |
| 长期满意度 | 用户长期参与度 |

### 6.3 性能提升

> "We demonstrate the efficacy of our approaches through a series of simulations and multiple live experiments on YouTube."

**翻译**：我们通过一系列模拟和YouTube上的多次在线实验证明了我们方法的有效性。

**关键成果**：
- REINFORCE推荐器成功扩展到极大动作空间
- Off-policy correction有效处理数据偏差
- Top-K off-policy correction显著提升Top-K推荐性能
- 探索策略发现新的有吸引力物品

---

## 七、与Survey论文的联系

### 7.1 作为Policy-based方法的实践案例

本论文是Survey论文中提到的"Policy-based方法"和"REINFORCE算法"的具体实践案例，展示了REINFORCE在工业推荐系统中的成功应用。

### 7.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| REINFORCE算法 | 策略梯度方法，扩展到极大动作空间 |
| Off-policy learning | 重要性采样修正，一阶近似 |
| 探索策略 | 策略中的随机性，发现新物品 |
| 状态表示 | RNN (CFN) 建模用户状态演化 |
| 动作表示 | 双重嵌入：$u_a$ 和 $v_a$ |
| Top-K推荐 | Novel Top-K off-policy correction |

### 7.3 创新点验证

| Survey提出的挑战 | 本论文解决方案 |
|----------------|--------------|
| 极大动作空间 | Sampled softmax + 最近邻搜索 |
| 数据偏差 | Off-policy correction + 行为策略建模 |
| Top-K推荐 | Novel Top-K off-policy correction |
| 探索-利用平衡 | 策略随机性 + 探索价值验证 |

### 7.4 与DRN论文的对比

| 维度 | DRN (Value-based) | YouTube REINFORCE (Policy-based) |
|------|-------------------|----------------------------------|
| 核心算法 | DQN | REINFORCE |
| 稳定性 | 需大量调参 | 函数近似稳定 |
| Off-policy | 天然支持 | 需要重要性采样修正 |
| 动作空间 | 中等规模 | 极大规模（数百万） |
| 推荐形式 | 单物品 | Top-K物品 |
| 探索策略 | DBGD | 策略随机性 |

---

## 八、工程实践启示

### 8.1 计算效率优化

**训练阶段**：
- Sampled softmax替代完整softmax
- 减少计算复杂度从 $O(|A|)$ 到 $O(K)$

**服务阶段**：
- 高效最近邻搜索算法
- 仅用Top动作近似softmax概率

### 8.2 数据处理

**日志数据**：
- 来自多个历史策略的混合
- 需要建模行为策略 $\beta$
- 使用神经网络学习 $\beta(a|s)$

### 8.3 在线实验

**A/B测试**：
- 真实用户环境验证
- 长期满意度指标
- 探索价值验证

---

## 九、论文引用

```
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi
Top-K Off-Policy Correction for a REINFORCE Recommender System
WSDM '19, February 11-15, 2019, Melbourne, VIC, Australia
```
