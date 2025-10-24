# 外层循环：超级BN（Hybrid Bayesian Network）设计方案

## 0. 目录结构说明

### 0.1 架构决策：分离引擎与知识

本设计采用**职责分离原则**，将外层循环的实现分布在三个独立模块：

```
src/
├── meta_model/        # 外层循环引擎（SuperBN, HybridBN）
│   ├── nodes.py           # 节点定义、类型枚举
│   ├── hybrid_bn.py       # 混合贝叶斯网络核心
│   ├── super_bn.py        # 高层API（面向科学发现任务）
│   └── inference.py       # 推理引擎（MCMC/VI）
│
├── knowledge_base/    # 知识存储与管理（LLM先验、文献）
│   ├── priors.py          # 先验分布管理
│   ├── literature.py      # 文献解析（LangChain集成）
│   ├── llm_integration.py # LLM API包装
│   └── domain_knowledge/  # 特定领域的知识库
│
├── acquisition/       # 主动学习策略（独立模块）
│   ├── strategies.py      # 采样策略基类
│   ├── uncertainty.py     # 不确定性采样
│   └── boundary_exploration.py  # 边界探索
│
└── ...
```

### 0.2 设计原则

1. **meta_model/**：纯粹的算法实现
   - 不依赖具体的知识来源
   - 通过接口接收先验（来自knowledge_base或用户）
   - 职责：BN结构管理、推理、学习

2. **knowledge_base/**：知识的提取、存储、管理
   - 从文献/LLM提取因果先验
   - 管理领域知识（物理定律、约束）
   - 提供标准化的先验接口供meta_model使用

3. **acquisition/**：主动学习策略
   - 通过依赖注入使用meta_model和InnerLoopEngine
   - 不直接依赖具体实现，只依赖接口
   - 可复用于不同的元模型架构

### 0.3 模块间交互示例

```python
# knowledge_base提取先验
from knowledge_base.literature import LiteratureParser
from knowledge_base.priors import PriorBuilder

parser = LiteratureParser(llm_api="gpt-4")
causal_graph = parser.extract_causal_structure("Hooke's Law literature")
priors = PriorBuilder.from_causal_graph(causal_graph)

# meta_model使用先验构建BN
from meta_model.super_bn import SuperBN

super_bn = SuperBN(rules=[hookes_law, plastic_law])
super_bn.build_initial_structure(
    context_vars=["force", "displacement", "temperature"],
    priors=priors  # 来自knowledge_base
)

# acquisition使用meta_model进行主动学习
from acquisition.uncertainty import MaxUncertaintyStrategy

strategy = MaxUncertaintyStrategy(meta_model=super_bn)
next_experiment = strategy.recommend()
```

---

## 1. 概念框架

### 1.1 超级BN的定义
**超级BN（Hybrid Bayesian Network）** 是同时包含离散变量和连续变量的贝叶斯网络，能够建模复杂的混合因果关系。

在Aetherium中：
- **输入**：上下文参数（连续或离散）+ 内层循环的likelihood scores
- **输出**：P(Rule适用性 | Context1, Context2, ...) 的联合分布
- **目标**：回答"WHEN does this law apply?"和"WHY does it fail?"

### 1.2 与内层循环的接口
```
内层循环（RuleClass）
   ↓
FitResult.log_likelihood (单个标量值)
   ↓
外层循环（SuperBN）
   ↓
Meta-model: P(适用性 | 上下文特征)
```

关键接口：`RuleClass.get_applicability_score(X, y, context) → float`

---

## 2. 技术栈选择

### 2.1 pgmpy的能力与限制

**pgmpy 支持**：
- ✅ 离散贝叶斯网络（Discrete BN）
- ✅ 高斯贝叶斯网络（Linear Gaussian BN）：连续变量 + 线性高斯关系
- ✅ 结构学习（score-based, constraint-based）
- ✅ 参数学习（MLE, Bayesian估计）
- ✅ 推理（Variable Elimination, Belief Propagation）

**pgmpy 限制**：
- ❌ **不直接支持混合离散-连续节点的任意条件分布**
- ❌ 不支持非线性连续关系（如 y = x² + noise）
- ❌ 不支持条件线性高斯（CLG）网络的自动推理

### 2.2 实现策略：混合方法

采用**分层架构**：

| 层级 | 技术 | 用途 |
|------|------|------|
| **结构层** | pgmpy | DAG定义、结构学习、因果推理 |
| **参数层** | PyMC / 自定义 | 混合节点的条件分布（CLG、非线性） |
| **推理层** | MCMC / VI | 近似贝叶斯推理（处理混合分布） |

---

## 3. 核心架构设计

### 3.1 三层抽象

```
SuperBN (最上层API)
   ↑
HybridBayesianNetwork (混合分布管理)
   ↑
pgmpy.BayesianNetwork (DAG结构)
```

### 3.2 类设计

#### 3.2.1 节点类型枚举
```python
class NodeType(Enum):
    DISCRETE = "discrete"           # 离散变量（如"regime: elastic/plastic"）
    CONTINUOUS = "continuous"       # 连续变量（如"velocity", "temperature"）
    LIKELIHOOD = "likelihood"       # 来自内层循环的log-likelihood
    APPLICABILITY = "applicability" # 规则适用性（目标变量）
```

#### 3.2.2 条件分布类型
```python
class ConditionalDistributionType(Enum):
    CATEGORICAL = "categorical"                 # P(discrete | parents)
    LINEAR_GAUSSIAN = "linear_gaussian"         # P(y | x) = N(α + βx, σ²)
    CONDITIONAL_LINEAR_GAUSSIAN = "clg"         # P(y | x, discrete_parent)
    NONLINEAR_GAUSSIAN = "nonlinear_gaussian"   # P(y | x) with PyMC model
    MIXTURE = "mixture"                          # 混合分布
```

#### 3.2.3 核心类

```python
@dataclass
class SuperBNNode:
    """超级BN中的节点定义"""
    name: str
    node_type: NodeType
    distribution_type: ConditionalDistributionType
    parents: List[str]

    # 离散变量：类别列表
    categories: Optional[List[str]] = None

    # 连续变量：分布参数或PyMC模型
    prior_params: Optional[Dict[str, Any]] = None
    pymc_model_fn: Optional[Callable] = None

    # 学习到的参数
    learned_params: Optional[Dict[str, Any]] = None

class HybridBayesianNetwork:
    """
    混合贝叶斯网络：pgmpy + PyMC的混合实现

    核心功能：
    1. 管理混合节点（离散+连续）
    2. 学习结构和参数
    3. 执行近似推理（MCMC/VI）
    4. 提供因果干预接口
    """

    def __init__(self):
        self.nodes: Dict[str, SuperBNNode] = {}
        self.dag: pgmpy.DAGModel = None
        self._pymc_model: Optional[pm.Model] = None

    def add_node(self, node: SuperBNNode) -> None:
        """添加节点到网络"""
        pass

    def add_edge(self, parent: str, child: str) -> None:
        """添加有向边（因果关系）"""
        pass

    def learn_structure(
        self,
        data: pd.DataFrame,
        algorithm: str = "hill_climbing",
        llm_priors: Optional[Dict] = None
    ) -> None:
        """
        学习BN结构

        支持算法：
        - hill_climbing: Score-based with BIC/BDeu
        - mmpc: Constraint-based (适合高维)
        - hybrid: 结合LLM先验 + 数据驱动
        """
        pass

    def learn_parameters(self, data: pd.DataFrame) -> None:
        """
        学习条件分布参数

        策略：
        - 纯离散节点：使用pgmpy的MLE/Bayesian估计
        - 纯连续（线性高斯）：使用pgmpy的LinearGaussianCPD
        - 混合节点：使用PyMC构建分层模型
        """
        pass

    def infer(
        self,
        evidence: Dict[str, Any],
        query_vars: List[str],
        method: str = "mcmc"
    ) -> Dict[str, np.ndarray]:
        """
        执行推理：P(query_vars | evidence)

        方法：
        - "exact": Variable Elimination (仅适用于离散+线性高斯)
        - "mcmc": MCMC采样（通用）
        - "vi": 变分推理（快速近似）
        """
        pass

    def do_intervention(
        self,
        intervention: Dict[str, Any],
        query_vars: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        因果干预：P(query | do(X=x))

        基于do-calculus，使用图手术（graph surgery）
        """
        pass

class SuperBN:
    """
    超级BN的高层API（面向Aetherium的科学发现任务）

    核心功能：
    1. 集成多个RuleClass的适用性评估
    2. 自动从数据+LLM先验构建BN
    3. 动态参数提升（Parameter Promotion）
    4. 规则切换（Rule Switching）
    """

    def __init__(self, rules: List[RuleClass]):
        self.rules = rules
        self.network = HybridBayesianNetwork()
        self.context_history: List[Dict] = []
        self.applicability_history: Dict[str, List[float]] = {}

    def build_initial_structure(
        self,
        context_vars: List[str],
        llm_priors: Optional[Dict] = None
    ) -> None:
        """
        构建初始BN结构

        策略：
        1. 从LLM提取的文献知识推断因果DAG
        2. 自动添加"适用性"节点作为子节点
        3. 将context_vars作为父节点
        """
        pass

    def update_with_experiment(
        self,
        context: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        用新实验数据更新BN

        流程：
        1. 对每个RuleClass调用fit()获取log_likelihood
        2. 将(context, likelihood)作为观测数据
        3. 更新BN的参数（增量学习）
        4. 返回各规则的适用性得分
        """
        pass

    def recommend_next_experiment(
        self,
        acquisition_strategy: str = "max_uncertainty"
    ) -> Dict[str, Any]:
        """
        主动学习：推荐下一个实验的上下文

        策略：
        - max_uncertainty: 选择BN最不确定的上下文区域
        - boundary_exploration: 探索规则边界（likelihood突变区域）
        - causal_identification: 选择最能识别因果关系的干预
        """
        pass

    def detect_failure_regions(
        self,
        rule: RuleClass,
        threshold: float = -10.0
    ) -> List[Dict[str, Any]]:
        """
        检测规则失效区域

        返回：低likelihood（< threshold）的上下文区域列表
        """
        pass

    def promote_parameter(
        self,
        rule: RuleClass,
        param_name: str,
        context_dependencies: List[str]
    ) -> None:
        """
        参数提升：将内层参数提升为上下文依赖

        示例：将Hooke's Law的k从常数提升为k = f(temperature, material)

        操作：
        1. 在BN中添加新节点"k"
        2. 添加边：context_dependencies → k
        3. 学习条件分布P(k | context)
        4. 更新RuleClass的prior
        """
        pass
```

---

## 4. 关键技术细节

### 4.1 条件线性高斯（CLG）的实现

对于节点 Y 有离散父节点 D 和连续父节点 X：

```
P(Y | D=d, X=x) = N(α_d + β_d · x, σ²_d)
```

**实现策略**：
```python
def build_clg_model(child: str, discrete_parent: str, continuous_parent: str, data: pd.DataFrame):
    with pm.Model() as model:
        # 为每个离散状态定义不同的线性关系
        discrete_states = data[discrete_parent].unique()

        # 切换参数
        alpha = pm.Normal(f"alpha_{child}", mu=0, sigma=10, shape=len(discrete_states))
        beta = pm.Normal(f"beta_{child}", mu=0, sigma=5, shape=len(discrete_states))
        sigma = pm.HalfNormal(f"sigma_{child}", sigma=2, shape=len(discrete_states))

        # 使用pm.switch根据离散状态选择参数
        state_idx = pm.Data("state_idx", data[discrete_parent].cat.codes.values)
        mu = alpha[state_idx] + beta[state_idx] * data[continuous_parent].values

        y_obs = pm.Normal("y", mu=mu, sigma=sigma[state_idx], observed=data[child].values)

    return model
```

### 4.2 非线性关系的处理

对于复杂的非线性依赖，使用PyMC的灵活性：

```python
def build_nonlinear_model(child: str, parents: List[str], data: pd.DataFrame):
    with pm.Model() as model:
        # 使用高斯过程或神经网络
        X = data[parents].values

        # 示例：使用Polynomial或GP
        lengthscale = pm.Gamma("lengthscale", alpha=2, beta=1)
        cov_func = pm.gp.cov.ExpQuad(input_dim=len(parents), ls=lengthscale)
        gp = pm.gp.Marginal(cov_func=cov_func)

        y_obs = gp.marginal_likelihood(child, X=X, y=data[child].values, noise=pm.HalfNormal("noise", sigma=1))

    return model
```

### 4.3 混合推理算法

```python
def hybrid_inference(network: HybridBayesianNetwork, evidence: Dict, query_vars: List[str]):
    """
    混合推理策略：
    1. 纯离散子图 → Variable Elimination (pgmpy)
    2. 混合/非线性部分 → MCMC (PyMC)
    3. 结果融合
    """
    # 分解网络
    discrete_vars, continuous_vars = network.partition_variables()

    # Step 1: 在离散子图上精确推理
    discrete_evidence = {k: v for k, v in evidence.items() if k in discrete_vars}
    discrete_marginals = network.pgmpy_network.query(
        variables=[v for v in query_vars if v in discrete_vars],
        evidence=discrete_evidence
    )

    # Step 2: 将离散推理结果作为条件，对连续变量采样
    with network.pymc_model:
        # 固定离散变量为推理结果的条件
        # 对连续query_vars进行MCMC采样
        trace = pm.sample(1000, tune=1000)

    # Step 3: 合并结果
    return {**discrete_marginals, **trace.posterior}
```

---

## 5. 演进路径

### Phase 1: 标准离散BN（快速原型）
**目标**：验证外层循环概念
- 离散化所有上下文变量（如 velocity: [low, medium, high]）
- 使用pgmpy的标准功能
- 简单的规则适用性分类

**优点**：实现快速，推理精确
**缺点**：丢失连续性信息，维度灾难

### Phase 2: 混合BN（当前设计）
**目标**：处理连续上下文 + 离散规则选择
- 上下文变量保持连续
- 适用性作为连续likelihood score或离散类别
- CLG模型 + PyMC后端

**优点**：保留完整信息，灵活建模
**缺点**：推理复杂度高，需要近似算法

### Phase 3: 结构因果模型（SCM）
**目标**：支持反事实推理和干预
- 显式因果机制：X → Y 对应确定性函数 Y = f(X) + noise
- 支持do-calculus和反事实查询
- 与符号发现（PySR）集成

**优点**：更强的解释性，支持因果干预
**缺点**：需要更强的因果假设

---

## 6. 集成示例

### 6.1 完整工作流

```python
# Step 1: 定义规则和上下文
hookes_law = LinearRule(metadata=RuleMetadata(name="Hooke's Law", formula="F=kx"))
plastic_law = BlackBoxRule(metadata=RuleMetadata(name="Plastic Deformation"))

rules = [hookes_law, plastic_law]
context_vars = ["force", "displacement", "material_type", "temperature"]

# Step 2: 初始化SuperBN
super_bn = SuperBN(rules=rules)

# 使用LLM先验构建初始结构
llm_priors = {
    "edges": [
        ("material_type", "applicability_hooke"),
        ("force", "applicability_hooke"),
        ("temperature", "applicability_plastic")
    ],
    "forbidden_edges": [("displacement", "temperature")]  # 物理约束
}
super_bn.build_initial_structure(context_vars=context_vars, llm_priors=llm_priors)

# Step 3: 主动学习循环
for iteration in range(100):
    # 推荐下一个实验
    next_context = super_bn.recommend_next_experiment(acquisition_strategy="max_uncertainty")

    # 执行实验（真实或仿真）
    X, y = run_experiment(next_context)

    # 更新BN
    applicability_scores = super_bn.update_with_experiment(next_context, X, y)

    # 检测失效区域
    failure_regions = super_bn.detect_failure_regions(hookes_law, threshold=-10.0)

    if failure_regions:
        print(f"Hooke's Law failing in: {failure_regions}")
        # 触发参数提升或规则发现

# Step 4: 查询推理
evidence = {"material_type": "steel", "temperature": 25.0, "force": 100.0}
applicability = super_bn.network.infer(
    evidence=evidence,
    query_vars=["applicability_hooke", "applicability_plastic"],
    method="mcmc"
)

print(f"Hooke's Law applicability: {applicability['applicability_hooke'].mean():.2f}")
print(f"Plastic Law applicability: {applicability['applicability_plastic'].mean():.2f}")

# Step 5: 因果干预
intervention = {"temperature": 200.0}  # 反事实：如果温度是200度
counterfactual = super_bn.network.do_intervention(
    intervention=intervention,
    query_vars=["applicability_hooke"]
)
```

---

## 7. 实现优先级

### 7.1 MVP（最小可行产品）

| 优先级 | 模块 | 功能 |
|--------|------|------|
| P0 | HybridBayesianNetwork | 基本节点管理、手动结构定义 |
| P0 | SuperBN | update_with_experiment（批量学习） |
| P0 | 推理引擎 | MCMC-based推理（纯PyMC实现） |

### 7.2 后续扩展

| 优先级 | 模块 | 功能 |
|--------|------|------|
| P1 | 结构学习 | hill_climbing + LLM先验 |
| P1 | 主动学习 | recommend_next_experiment |
| P2 | 参数提升 | promote_parameter |
| P2 | 因果推理 | do_intervention |
| P3 | 符号发现集成 | 自动触发PySR |

---

## 8. 技术挑战与解决方案

### 8.1 挑战：高维上下文空间（10,000+变量）

**解决方案**（来自CLAUDE.md）：
1. **多阶段过滤**：
   - 使用MMPC算法找Markov Blanket → 130候选变量
   - 合并LLM先验
   - 在精简子集上运行精确结构学习

2. **先验正则化**：
   - 初始化权重矩阵W时使用LLM先验
   - 损失函数加入先验权重：`λ Σ(αᵢⱼ · |Wᵢⱼ|)`

### 8.2 挑战：混合推理的计算复杂度

**解决方案**：
1. **分层推理**：先精确推理离散部分，再条件采样连续部分
2. **变分推理**：使用ADVI加速（牺牲一些精度）
3. **缓存机制**：相似上下文复用推理结果

### 8.3 挑战：动态结构变化（参数提升）

**解决方案**：
1. 使用**版本化BN**：每次结构变更创建新版本
2. 增量学习：仅重新学习受影响的局部结构
3. 迁移学习：将旧BN的参数作为新BN的先验

---

## 9. 测试策略

### 9.1 单元测试

```python
def test_hybrid_network_creation():
    """测试混合网络的基本创建"""
    network = HybridBayesianNetwork()

    # 添加离散节点
    discrete_node = SuperBNNode(
        name="regime",
        node_type=NodeType.DISCRETE,
        distribution_type=ConditionalDistributionType.CATEGORICAL,
        parents=[],
        categories=["elastic", "plastic"]
    )
    network.add_node(discrete_node)

    # 添加连续节点
    continuous_node = SuperBNNode(
        name="force",
        node_type=NodeType.CONTINUOUS,
        distribution_type=ConditionalDistributionType.LINEAR_GAUSSIAN,
        parents=["regime"],
        prior_params={"mu": 0, "sigma": 10}
    )
    network.add_node(continuous_node)

    assert len(network.nodes) == 2
    assert network.dag is not None

def test_clg_inference():
    """测试条件线性高斯推理"""
    # 模拟数据：Y = α[regime] + β[regime] * X + noise
    # 使用合成数据测试推理精度
    pass
```

### 9.2 集成测试

使用**合成物理系统**：
- Hooke's Law (F=kx) 在小位移下有效
- 塑性变形在大位移下出现
- 温度影响材料刚度k

验证：
1. SuperBN能否学到正确的边界条件
2. 参数提升能否发现k与temperature的依赖
3. 主动学习能否高效探索失效边界

---

## 10. 下一步行动

### 立即开始实现：

1. **创建基础类**（src/knowledge_base/）:
   - `nodes.py`: SuperBNNode, NodeType, ConditionalDistributionType
   - `hybrid_bn.py`: HybridBayesianNetwork
   - `super_bn.py`: SuperBN（高层API）

2. **实现MVP功能**：
   - 手动构建简单的3节点网络（1离散 + 2连续）
   - PyMC-based推理引擎
   - 与现有LinearRule集成

3. **编写测试**：
   - tests/cicd/knowledge_base/test_hybrid_bn.py
   - 使用Hooke's Law + 合成数据

是否开始实现代码？
