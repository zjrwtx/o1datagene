# Long Thought Chain Generator

这是一个使用LLM（大型语言模型）生成长思维链的工具，它结合了多种方法来模拟人类的思考过程，包括反思、回溯和经验总结。

## 功能特点

1. **树搜索方法**：将推理建模为树搜索过程，通过回溯找到正确解决方案
2. **提议-批评循环**：通过预定义的动作（继续、回溯、反思、终止）构建推理树
3. **多智能体方法**：使用策略模型和批评模型进行对话式推理
4. **人类思维过程注释**：模拟人类解决问题的思维过程

## 安装

1. 克隆项目并安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
- 复制 `.env.example` 为 `.env`
- 在 `.env` 文件中设置你的 DeepSeek API key

## 使用方法

```python
from thought_chain_generator import ThoughtChainGenerator

# 创建生成器实例
generator = ThoughtChainGenerator()

# 准备问题（可以是代码或数学问题）
problem = """
你的问题内容
"""

# 生成思维链
thought_chain = generator.generate_comprehensive_thought_chain(problem)

# 结果会保存在 thought_chain_output.json 文件中
```

## 输出示例

生成的思维链将包含以下部分：
- 输入问题
- 树搜索结果
- 提议-批评循环结果
- 多智能体对话结果
- 人类思维过程注释

## 注意事项

- 需要有效的 DeepSeek API key
- 确保网络连接正常
- 根据API限制调整请求频率
