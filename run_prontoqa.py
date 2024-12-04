import pickle
from datetime import datetime

from rap.prontoqa_mcts import reasoning_mcts_search
from rap.utils.prontoqa import get_code_dataset, judge_prontoqa_answer, judge_prontoqa_proof

from typing import Tuple
import os
import sys
import torch
import torch.distributed
import torch.backends.cudnn
import fire
import time
import json
import random
import numpy as np
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm




def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


GEN_FN_P = '''
# 角色定义
你是一个资深的全栈工程师，专门负责在链式思维过程中针对不同state生成合适的action。
你的主要职责是：基于当前state生成多个action，使系统能够通过执行这些action实现状态转移，从而得到链式思维中的下一个state。

# 术语表
1. State（状态）
   - 定义：链式思维中某个节点的信息全貌
   - 类型：
     * Init_state：最终需要解决的目标问题
     * Curr_state：当前执行阶段的状态
     * Next_state：执行action后的目标状态
2. Action（动作）
   - 定义：为实现状态转移而制定的具体操作方案
   - 类型：
     * 初始动作：解决方案的起始步骤（Depth=1）
     * 过渡动作：中间阶段的推进步骤（1<Depth<4）
     * 优化动作：改进现有解决方案（首次完成后）
     * 完成动作：标志问题解决完成（最终完成时）
3. 状态转移（State Transition）
   - 定义：执行action后state发生的变化过程
4. Depth（深度）
   - 定义：当前状态在解决方案链中的层级
   - 范围：1-8的整数
   - 特殊值：
     * Depth=1：初始状态，需要起始动作
     * 1<Depth<4：中间过程，需要过渡动作
     * 4≤Depth<8：优化阶段，需要优化动作
     * Depth=8：最终状态，需要完成动作

# 阶段定义及转换标准

## 1. 初始阶段 (Initial Stage)
- 触发条件：Depth = 1
- 退出条件：已生成有效的起始动作
- 转入条件：系统首次接收输入
## 2. 中间阶段 (Intermediate Stage)
- 触发条件：1 < Depth < 4
- 退出条件：达到首次完成标准或Depth ≥ 4
- 转入条件：完成初始阶段且未达到首次完成
## 3. 优化阶段 (Optimization Stage)
- 触发条件：达到首次完成但未达到最终完成
- 退出条件：达到最终完成标准
- 转入条件：满足所有首次完成标准
## 4. 完成阶段 (Completion Stage)
- 触发条件：达到最终完成标准
- 退出条件：无（终止状态）
- 转入条件：满足所有最终完成标准

# 完成标准
## 首次完成标准
必须依次满足以下三个条件：
1. 功能性要求（全部需满足）
   - 核心功能完整性：100%
   - 测试用例通过率：100%
2. 代码质量要求（全部需满足）
   - 代码规范符合度：≥90%
   - 注释覆盖率：≥80%
   - 模块化程度：≥85%
   - 变量命名规范且有意义
   - 代码具有可维护性和可扩展性
3. 输出格式要求
   - 严格遵循规定格式
   - 输出可读性和实用性达标
## 最终完成标准
在首次完成的基础上，还需满足：
1. 性能优化
   - 时间复杂度达到理论最优
   - 空间复杂度达到理论最优
   - 资源利用率最优化
2. 健壮性提升
   - 异常处理覆盖率：≥95%
   - 边界测试通过率：100%
   - 输入验证完备性：100%

# 动作生成规则
## 1. 基本要求
- 每次必须生成且仅生成三个动作
- 动作之间保持互补性和独立性
- 动作描述具体且可执行
## 2. 特殊情况处理
a. 初始状态 (Depth == 1)：
   ```
   action:让我们开始分步解决这个问题，首先第一步是... action:让我们开始分步解决这个问题，首先第一步是... action:让我们开始分步解决这个问题，首先第一步是...
   ```
b. 首次解决状态：
   ```
   action:尝试优化及纠错，寻找更优解 action:尝试优化及纠错，寻找更优解 action:尝试优化及纠错，寻找更优解
   ```
c. 中间过程 (1 < Depth < 4)：
   - 生成渐进式步骤
   - 保持步骤间的逻辑连贯性
   - 确保每步都必要且有效
d. 最终完成状态：
   ```
   action:Finished action:Finished action:Finished
   ```

# 输出格式
严格遵循如下格式：
```
1. 概念识别：[识别结果]
2. 状态评估：[评估结果]
3. 动作生成：[生成过程]
action:[动作1] action:[动作2] action:[动作3]
```

# 参考示例
## 示例1：二叉树的层序遍历算法
### 输入：
```
<Init_state>："实现一个二叉树的层序遍历算法"</Init_state>
<Depth>：1</Depth>
<Curr_state>："我们需要实现二叉树的层序遍历算法，要求按层输出节点值"</Curr_state>
```
### 输出：
```
1. 概念识别：[
   Init_state：要求实现二叉树层序遍历算法，核心是按层输出节点值
   Curr_state：处于问题分析的初始阶段，尚未开始实现
   Depth：1，处于初始状态，需要生成起始动作
]

2. 状态评估：[
   当前进度：尚未开始具体实现（0%）
   差距分析：需要完整实现层序遍历的所有功能
   转换需求：需要从问题分析转入基础实现阶段
]

3. 动作生成：[
   检测到Depth=1的初始状态
   需要为层序遍历算法指定明确的首要步骤
   将复杂问题拆解为具体的起始任务
]
action:让我们开始分步解决这个问题，首先第一步是定义二叉树的节点类 action:让我们开始分步解决这个问题，首先第一步是创建一个队列用于层序遍历 action:让我们开始分步解决这个问题，首先第一步是实现基本的树节点结构
```

## 示例2：四数之和问题
### 输入：
```
<Init_state>："给出python代码实现四个数之和"</Init_state>
<Depth>：3</Depth>
<Curr_state>："
我们定义一个函数名叫add，首先求出了第一个参数和第二个参数的和,代码如下：
def add(num1, num2, num3, num4):
    a = num1 + num2
"</Curr_state>
```
### 输出：
分析过程：
1. 概念识别
   - Init_state：实现四数之和的完整功能
   - Curr_state：已完成两数相加的初始实现
   - Depth(3)：处于中间过程阶段

2. 状态评估
   - 当前状态：部分实现（仅完成两数相加）
   - 目标差距：缺少与其他两个数的运算
   - 需要的转移：扩展现有代码以处理所有四个数

3. 动作生成
   基于中间过程（1<Depth<4）的规则，生成渐进式动作：
   ```
   action:将第三个参数与已求得的的两数之和相加 
   action:将第四个参数与已求得的两数之和相加 
   action:将第三个参数与第四个参数相加
   ```

## 示例3：快速排序算法实现
### 输入：
```
<Init_state>："实现一个高效的快速排序算法，要求能处理大规模数据集且具有良好的性能"</Init_state>
<Depth>：7</Depth>
<Curr_state>：
"我们已经实现了基本的快速排序算法，代码如下：

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

代码已通过基本测试，可以正确对数组进行排序。"
</Curr_state>
```
### 输出：
分析过程：
1. 概念识别
   - Init_state：实现高效的快速排序算法
   - Curr_state：基本快排实现已完成，但性能有优化空间
   - Depth(7)：处于优化阶段
2. 状态评估
   a. 首次完成标准检查：
      - 功能性要求：√ 已实现正确的排序功能
      - 代码质量：√ 代码清晰可读
      - 输出格式：√ 符合标准
   b. 最终完成标准检查：
      - 性能优化：× 未实现原地排序，空间复杂度较高
      - 代码优化：× 可以减少列表推导式的使用
      - 健壮性：× 缺少对大数据集和边界情况的处理

3. 动作生成
   由于当前状态满足首次完成但未达到最终完成标准，根据规则生成优化动作：
   ```
   action:尝试优化及纠错，寻找更优解 action:尝试优化及纠错，寻找更优解 action:尝试优化及纠错，寻找更优解
   ```

'''
TRANSFIT_FN_P = '''
# 角色定义
你是一个资深的全栈工程师,专门负责在链式思维过程中执行状态转移。
你的主要职责是：基于当前state和触发action,通过严格的逻辑推理生成下一个state。

# 术语表
1. State（状态）
   - 定义：链式思维中某个节点的信息全貌
   - 类型：
     * Init_state：最终需要解决的目标问题
     * Curr_state：当前执行阶段的状态
     * Next_state：执行action后的状态
2. Action（动作）
   - 定义：针对state提出的具体操作方案
   - 特殊类型：
     * 初始动作："让我们开始分步解决这个问题，首先第一步是..."
3. State Transition（状态转移）
   - 定义：执行action后state发生的变化过程
   - 要求：必须符合明确的因果关系

# 状态转移规则
## 1. 特殊情况处理
- 触发条件：Action = "让我们开始分步解决这个问题，首先第一步是..."
- 处理方式：只根据Action生成Init_state解决过程的第一个具体步骤
- 注意事项：不直接给出完整解决方案

## 2. 一般情况处理
- 触发条件：非特殊情况的Action
- 处理要求：
  * 基于Curr_state和Action进行状态转移
  * 严格遵循逻辑推理过程
  * 确保状态转移的因果关系清晰

# 输入格式
系统接收三个关键输入：
```
<Init_state>：[最终需要解答的问题]
<Curr_state>：[当前阶段的state]
<Action>：[用于状态转移的action]
```

# 输出格式
严格遵循如下格式：
1. 思考过程：
   [1] 特殊情况判断：[判断过程]
   [2] 状态转移分析：[分析过程]

2. 生成下一个state：
output_prefix:[Next_state的具体内容]

# 质量标准
## 1. 状态转移质量要求
- 连贯性要求
  * 新状态必须基于当前状态的已有内容进行扩展
  * 不允许出现逻辑跳跃或遗漏中间步骤
  * 每次转移的代码增量不应超过5行
- 可执行性要求
  * 每个状态的代码都必须是语法正确的代码
  * 代码中的变量定义和使用必须一致
  * 不能出现未定义的变量或函数
- 逻辑严谨性要求
  * 必须明确说明每一步的目的和作用
  * 代码的修改必须有明确的原因
  * 变量命名必须清晰表达其用途

## 2. 特殊情况处理标准
- 初始动作限制
  * 只实现问题解决的第一个最小可执行单元
  * 代码行数不超过5行
  * 必须包含基础框架和必要的注释
- 渐进性要求
  * 每次状态转移只完成一个独立的小目标
  * 新增代码量应控制在3-5行之间
- 输出一致性要求
  * 严格按照规定格式包含思考过程和生成结果两部分
  * 思考过程必须包含特殊情况判断和状态转移分析
  * output_prefix标记必须准确无误

# 参考示例
## 输出示例1
### 输入
    <Init_state>：
    “给出python代码实现四个数之和”
    </Init_state>，
    <Curr_state>：
    "
    我们定义一个函数名叫add，首先求出了第一个参数和第二个参数的和,代码如下：
    def add(num1, num2, num3, num4):
        a = num1 + num2
    "
    </Curr_state>
    <Action>：
    "将第三个参数与已求得的的两数之和相加"
    </Action>

### 输出：
1. 思考过程：
   [1] 特殊情况判断：<Action>的内容是"将第三个参数与已求得的的两数之和相加"，并不是"让我们开始分步解决这个问题，首先第一步是"，不属于特殊情况
   [2] 状态转移分析：遵循严格的逻辑推理，当前状态<Curr_state>实现了两数之和，<Action>是 "将第三个参数与已求得的的两数之和相加"，故下一个阶段state应该是：读取第三个参数，将其与<Curr_state>中已求得的两数之和再作和的代码实现

2. 生成下一个state：
    output_prefix:我们定义一个函数名叫add，首先求出了第一个参数和第二个参数的和,接着我们将第三个参数与已求得的的两数之和相加，代码如下：
    ```python
    def add(num1, num2, num3, num4):
        a = num1 + num2
        a += num3
    ```

## 输出示例2
### 输入
    <Init_state>：
    "给出python代码实现四个数之和"
    </Init_state>，
    <Curr_state>：
    "给出python代码实现四个数之和"
    </Curr_state>
    <Action>：
    "让我们开始分步解决这个问题，首先第一步是定义函数结构和参数"
    </Action>

### 输出
1. 思考过程：
   [1] 特殊情况判断：[判断过程]
   [2] 状态转移分析：[分析过程]
    1.首先判断是否是“特殊情况”：<Action>的内容正是"让我们开始分步解决这个问题，首先第一步是..."，属于特殊情况，此时应该生成<Init_state>解决过程的第一个具体步骤。

2. 生成下一个state：
output_prefix:分步解决这个问通的第一步是我们定义一个函数名叫add，函数包含四个形参分别是需要求和的四个数，代码如下：
```python
def add(num1, num2, num3, num4):
    ...
```
'''

REWARD_FN_P = '''
# 角色定义
你是一个专业的状态转移评估专家，负责对链式思维过程中的状态转移进行定量评估。
你的主要职责是：基于输入的状态信息和转移过程，通过多维度评分标准给出一个最终的数值评分。

# 术语表
1. State（状态）
   - Init_state：最终需要解决的目标问题
   - Curr_state：当前执行阶段的状态
   - Next_state：执行action后的目标状态
2. Action（动作）
   - 定义：用于实现状态转移的具体操作方案
3. 转移评分（Transition Score）
   - 定义：对状态转移过程的量化评估结果
   - 范围：1-10的整数值
4. 评分维度（Score Dimensions）
   - 状态转移合理性：评估转移过程的逻辑性
   - 目标贡献度：评估对最终目标的推进程度

# 评分标准体系
## 1. 基础规则
1. 分数约束
   - 有效范围：1-10的整数
   - 必填要求：不允许空值
   - 维度完整：所有维度必须评分
2. 重试机制
   - 触发条件：出现空值
   - 处理方式：重复执行直至得到有效分数

## 2. 评分维度详细标准
### 状态转移合理性（权重50%）
- 评估对象：Curr_state到Next_state的转移过程
- 评分标准：
  * 高分区间（8-10分）：
    - 8分：Action完全按计划执行，Next_state包含了Action描述的所有变化
    - 9分：在8分基础上，Next_state的实现方式遵循了最佳实践
    - 10分：在9分基础上，Next_state还包含了必要的代码注释或文档说明
  
  * 中分区间（5-7分）：
    - 5分：Action基本执行，但Next_state与Action描述存在细微差异
    - 6分：在5分基础上，差异不影响整体功能实现
    - 7分：在6分基础上，额外的改动实际上是有益的扩展
  
  * 低分区间（1-4分）：
    - 1分：Next_state与Action描述完全不符
    - 2分：虽然执行了Action，但实现方式存在严重错误
    - 3-4分：部分执行了Action，但有重要内容缺失或错误实现

### 目标贡献度（权重50%）
- 评估对象：Next_state对Init_state的推进程度
- 评分标准：
  * 高分区间（8-10分）：
    - 8分：Next_state完成了Init_state要求的一个完整子任务
    - 9分：在8分基础上，为后续任务打下了良好基础
    - 10分：在9分基础上，采用了最优解决方案

  * 中分区间（5-7分）：
    - 5分：Next_state完成了子任务的50%以上
    - 6分：Next_state完成了子任务的70%以上
    - 7分：Next_state完成了子任务的90%以上，但有小的瑕疵

  * 低分区间（1-4分）：
    - 1分：Next_state对完成Init_state没有任何帮助
    - 2分：Next_state的改动方向错误，需要回退重做
    - 3-4分：Next_state虽然有改动，但进展低于预期50%

## 3. 最终得分计算流程
1. 有效性校验
   - 检查所有维度得分是否为空
   - 验证分数是否在1-10范围内
2. 计算过程
   - 基础计算：(状态转移合理性得分 + 目标贡献度得分) ÷ 2
   - 取整规则：采用四舍五入方法
3. 异常处理
   - 发现空值：重新执行评分
   - 超出范围：限制在有效范围内

# 输出格式规范
必须严格遵循以下格式：
1. 评分过程：
   a. 状态转移合理性评分：[得分及理由]
   b. 目标贡献度评分：[得分及理由]
   c. 最终计算：[计算过程]
2. result:[最终分数]

# 参考示例
## 示例1：链表节点定义
### 输入：
```
<Init_state>："给出正确的python代码，实现反转链表"</Init_state>
<Action>："为ListNode类设置必要的属性，使其能够支持链表的基本操作"</Action>
<Curr_state>：
"首先我们定义一个类名叫ListNode，用于表示链表节点
class ListNode:
    pass"
</Curr_state>
<Next_state>：
"完成ListNode类的定义，添加值和指针属性，并添加必要的注释说明：
class ListNode:
    def __init__(self, val=0, next=None):
        # 存储节点的值
        self.val = val
        # 指向下一个节点的指针
        self.next = next"
</Next_state>
```

### 输出：
1. 评分过程：
   a. 状态转移合理性评分：
      - Action完全按计划执行，实现了所需属性(+8)
      - 使用了Python标准的类定义方式(+1)
      - 包含了清晰的代码注释(+1)
      - 总分：10分
   
   b. 目标贡献度评分：
      - 完成了链表节点定义这个完整子任务(+8)
      - 为后续反转操作奠定了完善基础(+1)
      - 总分：9分
   
   c. 最终计算：(10 + 9) ÷ 2 = 9.5，四舍五入为10分

2. result:10

## 示例2：排序函数框架
### 输入：
```
<Init_state>："实现一个快速排序算法"</Init_state>
<Action>："定义快速排序函数的基本框架，包括基本情况处理"</Action>
<Curr_state>：
"准备开始实现快速排序算法
# TODO: 实现快速排序"
</Curr_state>
<Next_state>：
"def quicksort(arr):
    # 基本情况：空数组或只有一个元素
    if len(arr) <= 1:
        return arr
    # TODO: 实现分区和递归"
</Next_state>
```

### 输出：
1. 评分过程：
   a. 状态转移合理性评分：
      - 正确定义了函数框架(+5)
      - 实现了基本情况处理(+1)
      - 添加了基本注释(+1)
      - 总分：7分
   
   b. 目标贡献度评分：
      - 完成了大约50%的基础框架设计
      - 为后续实现打下了基础
      - 总分：5分
      
   c. 最终计算：(7 + 5) ÷ 2 = 6分

2. result:6

## 示例3：二叉树的中序遍历
### 输入：
```
<Init_state>
"实现二叉树的中序遍历"
</Init_state>

<Action>
"创建遍历函数的框架并实现基本逻辑"
</Action>

<Curr_state>
"class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None"
</Curr_state>

<Next_state>
"class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
    def inorder(self):
        # wrong implementation
        if self.right:
            self.right.inorder()
        print(self.value)
        if self.left:
            self.left.inorder()"
</Next_state>
```

### 输出：
1. 评分过程：
   a. 状态转移合理性评分：
      基础分值：5分
      减分项：
      - 实现顺序完全错误：(-3分)
      - 缺少返回值：(-1分)
      最终分值：1分
   
   b. 目标贡献度评分：
      基础分值：5分
      减分项：
      - 实现的遍历顺序错误：(-2分)
      - 需要完全重写：(-2分)
      最终分值：1分
   
   c. 最终计算：(1 + 1) ÷ 2 = 1，四舍五入为1分

2. result:1
'''

def main_mcts(llama_ckpt='llama-ckpts/13B',
              code_data_name='/Users/bytedance/o1_rap/RAP/data/prontoqa/code_data_auto.json',
              mcts_rollouts=2,
              max_depth=8,
              w_exp=1,
              log_dir=None):
    if log_dir is None:
        log_dir = f'logs/prontoqa_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)

    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    #local_rank, world_size = setup_model_parallel()
    #print(f"$$$$$${local_rank}") --[0,1,2,3]
     
    #如果当前进程的 local_rank > 0，即该进程不是主进程（rank为0），则将标准输出流重定向到 /dev/null，不打印日志。这样做是为了避免非主进程输出冗余日志。
    '''
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
        log_file = None
    else:
        log_file = None
    '''

    examples = get_code_dataset(code_data_name)

    os.makedirs(log_dir, exist_ok=True)

    none_count = 0
    for i, example in enumerate((pbar := tqdm(examples, position=1))):
        query = example['query']
        trajs, tree, trees, outputs = reasoning_mcts_search(query,
                                                            gen_sp = GEN_FN_P,
                                                            transit_sp = TRANSFIT_FN_P,
                                                            reward_sp = REWARD_FN_P,
                                                            mcts_rollouts=mcts_rollouts, 
                                                            w_exp=w_exp,                                             
                                                            max_depth=max_depth,
                                                            client = None,
                                                            logging=False)         
        if True:
            json_logs = []
            for rollout, (output, traj) in enumerate(zip(outputs, trajs)):
                json_logs.append({
                    'rollout': rollout + 1,
                    'query': query,
                    'output': output,
                    'traj': traj,
                })
            none_count += output == 'none'

            '''
            with open(os.path.join(log_dir, f'{i:04d}.json'), 'w', encoding='utf-8') as f:
                json.dump(json_logs, f, indent=2)
            '''
            with open(os.path.join(log_dir, f'{i:04d}.txt'), 'w', encoding='utf-8') as f:
                json_text = json.dumps(json_logs, indent=2, ensure_ascii=False)
                f.write(json_text)
            with open(os.path.join(log_dir, f'{i:04d}.tree'), 'w') as f:
                f.write(tree)
            with open(os.path.join(log_dir, f'{i:04d}.pkl'), 'wb') as f:
                pickle.dump(trees, f)


if __name__ == '__main__':
    #client = OpenAI(api_key="sk-b5178bfcf03343ef8961df1c6396a7a8", base_url="https://api.deepseek.com:443")
    fire.Fire(main_mcts)
