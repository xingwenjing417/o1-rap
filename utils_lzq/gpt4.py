import openai
import time 
import tqdm

def setup_openai_client_gpt4():
    base_url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
    api_version = "2024-03-01-preview"
    ak = "cPwaQZNU657PaO8GzGszsCcYsCNM91Is"
    return openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )
def workflow_gpt4(userprompt, system_prompt, max_retries=20):
    client = setup_openai_client_gpt4()
    model_name = "gpt-4-turbo-2024-04-09" # 或者gpt-4o-2024-08-06或者o1-preview-2024-09-12
    max_tokens = 4096 # o1最大8000
    def make_api_request():
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": userprompt
                },
                {
                    "role": "system",
                    "content": system_prompt
                }
            ],
            max_tokens=max_tokens,
            extra_headers={"X-TT-LOGID": "${your_logid}"},
        )
        # Parse the response to get the content
        response_data = completion.choices[0].message.content
        return response_data
    # 实现重试逻辑
    for attempt in range(max_retries):
        try:
            return make_api_request()
        except Exception as e:
            print(f"Attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2.5)  # 重试前等待
                continue
            return None
def concurrent_apply_gpt4(user_in,sp):
    try:
        result = workflow_gpt4(str(user_in),str(sp))  # 确保转换为字符串
    except Exception as e:
        print(f"Processing error: {e}")
    return result

if __name__ == "__main__":
    sp = '''
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

    user_in = f"Init_state:用python实现三数之积,Curr_state:用python实现三数之积,Depth:1"
    out = concurrent_apply_gpt4(user_in, sp)
    print(out)
    