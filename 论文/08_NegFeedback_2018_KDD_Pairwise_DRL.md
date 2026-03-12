# 08_NegFeedback_2018_KDD_Pairwise_DRL

**论文信息**
- 标题：Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning
- 作者：Xiangyu Zhao, Liang Zhang, Zhuoye Ding, Long Xia, Jiliang Tang, Dawei Yin
- 单位：Michigan State University, JD.com
- 发表：KDD 2018

---

## 一、核心痛点与动机 (The "Why")

### 1.1 传统推荐系统的局限

> "Most existing recommender systems including collaborative filtering, content-based and learning-to-rank consider the recommendation procedure as a static process and make recommendations following a fixed greedy strategy."

**翻译**：大多数现有推荐系统（包括协同过滤、基于内容和学习排序）将推荐过程视为静态过程，采用固定的贪心策略进行推荐。

**核心问题**：
| 问题 | 描述 | 影响 |
|------|------|------|
| 静态策略 | 固定贪心策略 | 无法捕捉用户偏好的动态特性 |
| 短视优化 | 仅最大化即时奖励 | 忽略长期奖励 |
| 忽略负反馈 | 只关注正向反馈 | 丢失重要用户偏好信息 |

### 1.2 负反馈的重要性

> "Users also skip some recommended items during the recommendation procedure. These skipped items influence user's click/order behaviors, which can help us gain better understandings about users' preferences."

**翻译**：用户在推荐过程中也会跳过一些推荐物品。这些跳过的物品影响用户的点击/购买行为，可以帮助我们更好地理解用户偏好。

**关键洞察**：
- 负反馈（跳过）数量远大于正反馈
- 负反馈揭示用户不喜欢什么
- 负反馈影响用户后续行为

### 1.3 负反馈的挑战

> "The number of skipped items (or negative feedback) is typically far larger than that of positive ones. Hence, it is challenging to capture both positive and negative feedback since positive feedback could be buried by negative one."

**翻译**：跳过的物品（或负反馈）数量通常远大于正反馈。因此，同时捕捉正负反馈具有挑战性，因为正反馈可能被负反馈淹没。

**核心挑战**：
- 负反馈数量 >> 正反馈数量
- 直接合并会导致正反馈被淹没
- 需要专门的方法处理

---

## 二、核心创新点

### 2.1 DEERS框架：同时建模正负反馈

> "We propose a framework DEERS to model positive and negative feedback simultaneously."

**翻译**：我们提出了DEERS框架来同时建模正负反馈。

**核心思想**：
- 状态包含正反馈 $s^+$ 和负反馈 $s^-$
- 分别建模正负反馈的贡献
- 推荐与正反馈相似、与负反馈不相似的物品

### 2.2 Pairwise正则化：利用部分偏好顺序

> "This partial order naturally inspires us maximizing the difference of Q-values between Q(s2, a2) and Q(s5, a5)."

**翻译**：这种部分顺序自然地启发我们最大化Q(s2, a2)和Q(s5, a5)之间的Q值差异。

**核心洞察**：
- 同一类别中，用户点击某些物品、跳过其他物品
- 这揭示了用户对这些物品的偏好顺序
- 可以利用这种部分顺序信息

### 2.3 GRU捕捉顺序偏好

> "We introduce a RNN with Gated Recurrent Units (GRU) to capture users' sequential preference."

**翻译**：我们引入了一个带有门控循环单元(GRU)的RNN来捕捉用户的顺序偏好。

**优势**：
- 捕捉用户偏好的时序特性
- GRU比LSTM更适合推荐任务
- 有效建模状态转移

---

## 三、问题建模

### 3.1 MDP定义

**MDP五元组** $\langle S, A, P, R, \gamma \rangle$：

| 要素 | 定义 | 推荐系统含义 |
|------|------|--------------|
| $S$ | 状态空间 | 用户浏览历史 |
| $A$ | 动作空间 | 推荐物品 |
| $P$ | 转移概率 | 状态转移 $p(s_{t+1}|s_t, a_t)$ |
| $R$ | 奖励函数 | 用户反馈（跳过/点击/购买） |
| $\gamma$ | 折扣因子 | 未来奖励折扣 |

### 3.2 目标

**寻找最优策略** $\pi: S \rightarrow A$，最大化累积奖励：
$$\max_\pi \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]$$

### 3.3 Q值函数

**Bellman方程**：
$$Q^*(s, a) = \mathbb{E}_{s'}\left[ r + \gamma \max_{a'} Q^*(s', a') | s, a \right]$$

---

## 四、DEERS框架详解

### 4.1 基础DQN模型

**状态定义**：
$$s = \{i_1, ..., i_N\}$$

其中 $i_n$ 是用户最近点击/购买的物品。

**状态转移**：
- 用户跳过：$s' = s$
- 用户点击/购买：$s' = \{i_2, ..., i_N, a\}$

**问题**：无法处理负反馈

### 4.2 DEERS状态定义

**新状态定义**：
$$s = (s^+, s^-)$$

其中：
- $s^+ = \{i_1, ..., i_N\}$：用户最近点击/购买的物品
- $s^- = \{j_1, ..., j_N\}$：用户最近跳过的物品

**新状态转移**：
- 用户跳过：$s'^+ = s^+$，$s'^- = \{j_2, ..., j_N, a\}$
- 用户点击/购买：$s'^+ = \{i_2, ..., i_N, a\}$，$s'^- = s^-$

### 4.3 GRU状态表示

**GRU更新门**：
$$z_n = \sigma(W_z i_n + U_z h_{n-1})$$

**GRU重置门**：
$$r_n = \sigma(W_r i_n + U_r h_{n-1})$$

**候选隐藏状态**：
$$\hat{h}_n = \tanh[W i_n + U(r_n \cdot h_{n-1})]$$

**隐藏状态更新**：
$$h_n = (1 - z_n) h_{n-1} + z_n \hat{h}_n$$

**状态表示**：
- 正状态：$s^+ = h_N^+$
- 负状态：$s^- = h_N^-$

### 4.4 分离网络架构

**核心思想**：
> "The intuition behind this architecture is to recommend an item that is similar to the clicked/ordered items (left part), while dissimilar to the skipped items (right part)."

**翻译**：这种架构背后的直觉是推荐与点击/购买物品相似（左侧），同时与跳过物品不相似（右侧）的物品。

**架构设计**：
- 正输入：$(s^+, a)$
- 负输入：$(s^-, a)$
- 分离的前几层隐藏层
- 合并后的输出层

### 4.5 损失函数

**标准损失**：
$$L(\theta) = \mathbb{E}_{s, a, r, s'}\left[ \left( y - Q(s^+, s^-, a; \theta) \right)^2 \right]$$

其中 $y = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q(s'^+, s'^-, a'; \theta_p) | s^+, s^-, a]$

**梯度**：
$$\nabla_\theta L(\theta) = \mathbb{E}_{s, a, r, s'}\left[ \left( r + \gamma \max_{a'} Q(s'^+, s'^-, a'; \theta_p) - Q(s^+, s^-, a; \theta) \right) \nabla_\theta Q(s^+, s^-, a; \theta) \right]$$

---

## 五、Pairwise正则化详解

### 5.1 动机：部分偏好顺序

**观察**：
- 同一会话中，RA推荐同类别的多个物品
- 用户点击/购买其中一些，跳过其他
- 这揭示了用户对这些物品的偏好顺序

**示例**：
| 时间 | 物品 | 类别 | 用户行为 |
|------|------|------|----------|
| 1 | a1 | A | 跳过 |
| 2 | a2 | B | 点击 |
| 3 | a3 | C | 点击 |
| 4 | a4 | A | 跳过 |
| 5 | a5 | B | 跳过 |
| 6 | a6 | C | 跳过 |

**部分顺序**：
- 类别B：$a2$ 被点击，$a5$ 被跳过 $\Rightarrow Q(s_2, a2) > Q(s_5, a5)$
- 类别C：$a3$ 被点击，$a6$ 被跳过 $\Rightarrow Q(s_3, a3) > Q(s_6, a6)$

### 5.2 竞争物品定义

**定义**：
- 同一会话中同类别的物品
- 选择时间最接近的作为"竞争物品"

**示例**：
- $a2$ 的竞争物品是 $a5$（同类别B，时间最接近）
- $a3$ 的竞争物品是 $a6$（同类别C，时间最接近）

### 5.3 Pairwise正则化项

**目标**：最大化被点击物品与被跳过物品的Q值差异

**正则化项**：
$$L_{pair} = \max(0, \delta - Q(s^+, a^+) + Q(s^-, a^-))$$

其中：
- $a^+$：被点击的物品
- $a^-$：竞争物品（被跳过）
- $\delta$：期望的Q值差异阈值

**总损失**：
$$L_{total} = L(\theta) + \lambda L_{pair}$$

### 5.4 优势

**核心优势**：
- 利用隐式的用户偏好顺序
- 增强正负反馈的区分
- 提高推荐准确性

---

## 六、训练与服务策略

### 6.1 离线训练

**数据来源**：
- 用户行为日志
- 包含跳过、点击、购买行为

**训练流程**：
1. 从日志构建状态-动作-奖励序列
2. 训练DQN网络
3. 添加Pairwise正则化

### 6.2 物品召回机制

**目的**：减少计算代价

**方法**：
- 不遍历所有物品
- 召回相关物品子集
- 在子集中选择最优动作

### 6.3 在线服务

**流程**：
1. 获取用户当前状态 $(s^+, s^-)$
2. 召回候选物品
3. 计算每个候选物品的Q值
4. 选择Q值最高的物品推荐

---

## 七、实验结果

### 7.1 数据集

- 真实电商数据
- 用户行为日志

### 7.2 评估指标

| 指标 | 说明 |
|------|------|
| 点击率 (CTR) | 用户点击推荐物品的比例 |
| 转化率 | 用户购买推荐物品的比例 |
| 累积奖励 | 整个交互过程的累积效用 |

### 7.3 性能对比

**对比方法**：
- 基础DQN（仅正反馈）
- 传统推荐方法

**关键发现**：
> "The experimental results based on real-world e-commerce data demonstrate the effectiveness of the proposed framework."

**翻译**：基于真实电商数据的实验结果证明了所提框架的有效性。

**消融实验**：
- 验证负反馈的重要性
- 验证Pairwise正则化的有效性

---

## 八、与Survey论文的联系

### 8.1 作为负反馈处理的实践案例

本论文是Survey论文中提到的"负反馈处理"的具体实践案例，展示了如何有效利用负反馈提升推荐性能。

### 8.2 关键技术对应关系

| Survey概念 | 本论文实现 |
|-------------|--------------|
| 负反馈建模 | DEERS框架 $(s^+, s^-)$ |
| 部分偏好顺序 | Pairwise正则化 |
| 状态表示 | GRU顺序建模 |
| DQN | 分离网络架构 |

### 8.3 与其他论文的对比

| 维度 | DRN | DeepPage | DEERS |
|------|-----|----------|-------|
| 算法 | DQN | Actor-Critic | DQN + Pairwise |
| 负反馈 | 无 | 无 | 显式建模 |
| 正则化 | 无 | 无 | Pairwise |
| 状态表示 | 用户特征 | GRU + CNN | GRU + 正负分离 |

---

## 九、工程实践启示

### 9.1 负反馈收集

**关键步骤**：
1. 记录用户跳过行为
2. 构建负反馈状态 $s^-$
3. 与正反馈状态 $s^+$ 配对

### 9.2 网络架构设计

**分离架构优势**：
- 正负反馈独立建模
- 避免正反馈被淹没
- 更好地捕捉用户偏好

### 9.3 Pairwise正则化应用

**适用场景**：
- 同类别物品推荐
- 用户有明确的偏好顺序
- 需要区分相似物品

---

## 十、论文引用

```
Xiangyu Zhao, Liang Zhang, Zhuoye Ding, Long Xia, Jiliang Tang, Dawei Yin
Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning
KDD '18, August 19–23, 2018, London, United Kingdom
```
