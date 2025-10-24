# 开发日志与架构决策记录 (Development Log & ADR)

本文档记录Aetherium项目的关键开发决策、进度和上下文信息，供新的Claude Code实例或开发者快速了解项目状态。

---

## 快速导航

- **主要指导文档**: [CLAUDE.md](CLAUDE.md) - 项目概述、结构、技术栈
- **设计哲学**: [docs/DesignPhilosophy.md](docs/DesignPhilosophy.md) - 核心理念和双循环架构
- **外层循环设计**: [docs/outer_loop_super_bn_design.md](docs/outer_loop_super_bn_design.md) - SuperBN详细设计
- **代码入口**:
  - 内层循环（RuleClass）: [src/rules/base.py](src/rules/base.py)
  - 外层循环（SuperBN）: [src/meta_model/](src/meta_model/) ← **当前开发中**

---

## 当前状态 (2025-10-24)

### ✅ 已完成
- **Phase 1 - 内层循环基础**
  - ✅ `RuleClass` 抽象基类 (src/rules/base.py)
  - ✅ `LinearRule` 实现 (src/rules/linear_rule.py)
  - ✅ 测试套件 (tests/cicd/rules/, tests/opt/rules/)
  - ✅ 诊断工具 (src/utils/diagnostics.py)

### 🚧 进行中
- **Phase 1 - 外层循环基础**
  - ✅ 创建 src/meta_model/ 目录结构
  - ✅ 创建 src/meta_model/__init__.py
  - ⏳ **NEXT**: 实现 src/meta_model/nodes.py (节点定义)
  - ⏳ **NEXT**: 实现 src/meta_model/hybrid_bn.py (混合BN核心)
  - ⏳ **NEXT**: 实现 src/meta_model/super_bn.py (高层API)
  - ⏳ **NEXT**: 编写测试 tests/cicd/meta_model/

### 📋 待办 (按优先级)
1. 完成 meta_model/ MVP 实现
2. 实现 src/core/ (InnerLoopEngine 包装器)
3. 创建端到端集成测试
4. 实现 knowledge_base/ (LLM先验提取)
5. 实现 acquisition/ (主动学习)
6. 实现 discovery/ (符号回归)

---

## 架构决策记录 (ADR)

### ADR-001: 目录结构重构 (2025-10-24)

**背景问题**:
- 原CLAUDE.md定义 `src/knowledge_base/` 为"Outer loop engine (Bayesian Network / SCM)"
- 但 "knowledge_base" 这个名字更像是"存储知识的地方"，而非"引擎"
- 存在职责混淆：算法实现 vs. 知识管理

**讨论的方案**:

**方案1: 分离引擎与知识** ✅ **已采用**
```
src/
├── meta_model/        # 外层循环引擎（算法实现）
├── knowledge_base/    # 知识提取与存储（数据管理）
└── acquisition/       # 主动学习（独立模块）
```

**方案2: knowledge_base作为复合模块**
```
src/
└── knowledge_base/
    ├── meta_model/    # 子模块：BN引擎
    ├── priors/        # 子模块：先验管理
    └── domain/        # 子模块：领域知识
```

**方案3: 改名为outer_loop/**
```
src/
└── outer_loop/
    ├── bn/            # BN引擎
    ├── priors/        # 先验管理
    └── llm/           # LLM集成
```

**决策**: 采用 **方案1**

**理由**:
1. **职责清晰**: meta_model（算法）、knowledge_base（数据）、acquisition（策略）三者解耦
2. **可测试性**: meta_model 可以独立测试，不依赖LLM或文献解析
3. **依赖注入**: acquisition 通过接口使用 meta_model，可复用于不同架构
4. **命名直观**: "meta_model" = meta-cognitive model，准确反映其功能

**影响**:
- ✅ 更新 CLAUDE.md 项目结构说明
- ✅ 更新 docs/outer_loop_super_bn_design.md 第0节
- ✅ 创建 src/meta_model/ 目录
- ⏳ 未来需实现 knowledge_base/ 的接口标准

**相关讨论**: 用户在session中明确提出质疑并推动此决策

---

### ADR-002: SuperBN技术选型 (2025-10-24)

**背景**:
- 需要实现混合贝叶斯网络（Hybrid BN）来处理离散+连续变量
- pgmpy 不直接支持任意混合条件分布

**决策**: 采用 **分层架构**

```
SuperBN (高层API)
   ↓
HybridBayesianNetwork (混合分布管理)
   ↓
pgmpy (DAG结构) + PyMC (条件分布)
```

**理由**:
1. **利用pgmpy优势**: 结构学习、因果推理、DAG管理
2. **利用PyMC灵活性**: 任意条件分布（CLG、非线性、GP）
3. **演进路径清晰**: Standard BN → Hybrid BN → SCM
4. **保持模块化**: 可以单独测试DAG结构和条件分布

**关键技术点**:
- 条件线性高斯 (CLG): 用 `pm.switch` 实现离散状态切换
- 非线性关系: 用 PyMC 的 GP 或多项式
- 混合推理: 离散部分用 Variable Elimination，连续部分用 MCMC

**参考文档**: [docs/outer_loop_super_bn_design.md](docs/outer_loop_super_bn_design.md) 第2.2节

---

### ADR-003: 三种建模模式的设计 (2025-10-23)

**决策**: RuleClass 支持三种模式
- White-Box: 已知公式，参数估计
- Gray-Box: 已知约束，约束拟合
- Black-Box: 无先验，GP拟合

**理由**: 见 [docs/DesignPhilosophy.md](docs/DesignPhilosophy.md) 第2节

**关键设计**: 外层循环对建模模式无感知，只需要 log_likelihood

---

## 开发时间线

### 2025-10-23
- ✅ 实现 RuleClass 基类和 LinearRule
- ✅ 编写测试套件
- ✅ 创建诊断工具

### 2025-10-24
- ✅ 讨论并确定目录结构（ADR-001）
- ✅ 更新 CLAUDE.md 反映当前状态
- ✅ 创建 docs/outer_loop_super_bn_design.md
- ✅ 创建 src/meta_model/ 目录
- ✅ 创建 src/meta_model/__init__.py
- 🚧 **当前**: 准备实现 nodes.py

---

## 下一步行动计划

### 立即任务 (今日/本周)
1. **实现 src/meta_model/nodes.py**
   - 定义 `NodeType` 枚举
   - 定义 `ConditionalDistributionType` 枚举
   - 实现 `SuperBNNode` 数据类
   - 参考: [设计文档 §3.2](docs/outer_loop_super_bn_design.md)

2. **实现 src/meta_model/hybrid_bn.py (MVP版本)**
   - 基本节点管理（add_node, add_edge）
   - 手动结构定义（暂不实现自动结构学习）
   - 简单的参数学习（纯离散或纯连续）
   - MCMC-based 推理

3. **实现 src/meta_model/super_bn.py (简化版)**
   - 基本的 `update_with_experiment()` 方法
   - 集成多个 RuleClass
   - 批量学习逻辑

4. **编写测试**
   - tests/cicd/meta_model/test_nodes.py
   - tests/cicd/meta_model/test_hybrid_bn.py
   - 使用合成数据（Hooke's Law + 塑性变形）

### 短期目标 (1-2周)
- 完成 Phase 1 "Walking Skeleton"
- 端到端测试：RuleClass → meta_model → 适用性评估

### 中期目标 (1个月)
- 实现 knowledge_base/ (LLM集成)
- 实现 acquisition/ (主动学习)
- 进入 Phase 2 "Passive Scientist"

---

## 关键接口设计

### Inner Loop → Outer Loop
```python
# 内层循环提供
FitResult.log_likelihood: float

# 外层循环使用
SuperBN.update_with_experiment(
    context: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]  # 返回各规则的适用性得分
```

### Knowledge Base → Meta Model
```python
# knowledge_base 提供
@dataclass
class CausalPrior:
    edges: List[Tuple[str, str]]          # (parent, child)
    forbidden_edges: List[Tuple[str, str]]
    edge_probs: Dict[Tuple[str, str], float]  # P(edge exists)

# meta_model 使用
SuperBN.build_initial_structure(
    context_vars: List[str],
    priors: CausalPrior
)
```

### Meta Model → Acquisition
```python
# meta_model 提供
class SuperBN:
    def get_uncertainty(self, context: Dict) -> float:
        """返回BN在给定上下文的不确定性"""
        pass

# acquisition 使用
class UncertaintyStrategy:
    def __init__(self, meta_model: SuperBN):
        self.meta_model = meta_model

    def recommend_next(self) -> Dict[str, Any]:
        # 查询 meta_model 的不确定性
        pass
```

---

## 常见问题 (FAQ)

### Q: 新实例如何快速上手？
**A**: 按顺序阅读：
1. [CLAUDE.md](CLAUDE.md) - 了解项目概况
2. [DEVELOPMENT.md](DEVELOPMENT.md) (本文件) - 了解当前状态和决策
3. [docs/outer_loop_super_bn_design.md](docs/outer_loop_super_bn_design.md) - 了解当前任务的技术细节
4. 查看代码：[src/rules/base.py](src/rules/base.py) 了解已完成的部分

### Q: 为什么采用混合 pgmpy + PyMC？
**A**: 见 ADR-002。简言之：pgmpy处理结构，PyMC处理复杂分布，二者互补。

### Q: 为什么 knowledge_base/ 和 meta_model/ 分离？
**A**: 见 ADR-001。职责分离：一个管数据，一个管算法。

### Q: acquisition/ 应该放在哪里？
**A**: 独立模块，通过依赖注入使用 meta_model 和 InnerLoopEngine。

### Q: 当前最紧急的任务是什么？
**A**: 实现 src/meta_model/nodes.py，然后是 hybrid_bn.py 和 super_bn.py 的 MVP 版本。

---

## 技术债务与已知问题

### 当前无技术债务
项目处于早期阶段，代码质量良好。

### 设计待验证点
1. **CLG推理效率**: PyMC的MCMC是否足够快？可能需要考虑VI
2. **高维上下文**: MMPC过滤策略尚未实现，高维场景未验证
3. **参数提升**: 动态结构变更的实现细节待设计

---

## 版本历史

### v0.1.0-alpha (2025-10-24) - 当前版本
- ✅ RuleClass 框架
- ✅ LinearRule 实现
- 🚧 SuperBN 开发中

### v0.0.1 (2025-10-23) - 初始提交
- 项目初始化
- 基础文档

---

**最后更新**: 2025-10-24
**维护者**: 通过 CLAUDE.md 指导的 Claude Code
**状态**: 🚧 积极开发中
