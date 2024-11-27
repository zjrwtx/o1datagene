import gradio as gr
import json
import markdown2
from thought_chain_generator import ThoughtChainGenerator
import time

def format_response(response_data):
    """格式化API响应为美观的HTML"""
    if 'choices' in response_data and len(response_data['choices']) > 0:
        content = response_data['choices'][0]['message']['content']
        # 将Markdown转换为HTML
        html_content = markdown2.markdown(content)
        return html_content
    return "No response content"

def generate_thought_chain(input_text, methods, progress=gr.Progress()):
    """生成思维链并格式化输出"""
    generator = ThoughtChainGenerator()
    results = {}
    
    method_map = {
        "树搜索方法": "tree_search",
        "提议-批评循环": "propose_critique",
        "多智能体方法": "multi_agent",
        "人类思维过程注释": "human_annotation"
    }
    
    total_methods = len(methods)
    for i, method in enumerate(methods, 1):
        progress(i/total_methods, desc=f"正在使用{method}生成思维链...")
        method_key = method_map[method]
        
        if method_key == "tree_search":
            result = generator.tree_search_approach(input_text)
        elif method_key == "propose_critique":
            result = generator.propose_critique_loop(input_text)
        elif method_key == "multi_agent":
            result = generator.multi_agent_debate(input_text)
        elif method_key == "human_annotation":
            result = generator.human_thought_annotation(input_text)
            
        results[method] = format_response(result)
        time.sleep(0.5)  # 为了更好地展示进度
    
    # 构建输出HTML
    output_html = ""
    for method in methods:
        output_html += f"<div class='method-result'><h2>{method}</h2>{results[method]}</div>"
    
    return output_html

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 长思维链生成器
    
    这个工具使用多种方法来生成详细的思维过程。请选择您想要使用的方法：
    """)
    
    with gr.Row():
        input_text = gr.Textbox(
            label="输入",
            placeholder="请输入代码或问题...",
            lines=5
        )
    
    with gr.Row():
        methods = gr.CheckboxGroup(
            choices=["树搜索方法", "提议-批评循环", "多智能体方法", "人类思维过程注释"],
            label="选择生成方法",
            value=["树搜索方法"]  # 默认选中第一个方法
        )
    
    with gr.Row():
        submit_btn = gr.Button("生成思维链", variant="primary")
    
    # 使用单个HTML组件来显示所有结果
    output_display = gr.HTML()
    
    # 添加CSS样式
    gr.Markdown("""
    <style>
    .method-result {
        margin: 20px 0;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    .method-result h2 {
        color: #2c3e50;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
    }
    </style>
    """)
    
    submit_btn.click(
        generate_thought_chain,
        inputs=[input_text, methods],
        outputs=[output_display]
    )

if __name__ == "__main__":
    demo.launch()