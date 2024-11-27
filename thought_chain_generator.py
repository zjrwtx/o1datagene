import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
import re

load_dotenv()

class ThoughtNode:
    def __init__(self, content: str, type: str, confidence: float = 0.0):
        self.content = content
        self.type = type  # 'thought', 'action', 'reflection', 'conclusion'
        self.confidence = confidence
        self.children: List[ThoughtNode] = []
        self.feedback: Optional[str] = None

class ThoughtChainGenerator:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.memory_buffer: List[Dict] = []  # 存储历史思维过程
        self.feedback_history: List[Dict] = []  # 存储反馈历史

    def _call_llm(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        """改进的LLM调用，支持温度控制和错误重试"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False
                }
                response = requests.post(self.api_url, headers=self.headers, json=data)
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                continue

    def _extract_key_concepts(self, text: str) -> List[str]:
        """提取文本中的关键概念"""
        # 使用LLM提取关键概念
        messages = [
            {"role": "system", "content": "请提取以下文本中的关键概念，以逗号分隔："},
            {"role": "user", "content": text}
        ]
        response = self._call_llm(messages, temperature=0.3)
        if 'choices' in response:
            concepts = response['choices'][0]['message']['content'].split(',')
            return [concept.strip() for concept in concepts]
        return []

    def _generate_subtasks(self, problem: str) -> List[str]:
        """将复杂问题分解为子任务"""
        messages = [
            {"role": "system", "content": "请将以下问题分解为具体的子任务，每个子任务应该是可独立解决的："},
            {"role": "user", "content": problem}
        ]
        response = self._call_llm(messages, temperature=0.5)
        if 'choices' in response:
            tasks = response['choices'][0]['message']['content'].split('\n')
            return [task.strip() for task in tasks if task.strip()]
        return []

    def _evaluate_solution(self, problem: str, solution: str) -> Dict[str, Union[float, str]]:
        """评估解决方案的质量"""
        messages = [
            {"role": "system", "content": "请评估以下解决方案的质量，给出1-10的分数和具体反馈："},
            {"role": "user", "content": f"问题：{problem}\n解决方案：{solution}"}
        ]
        response = self._call_llm(messages, temperature=0.3)
        if 'choices' in response:
            content = response['choices'][0]['message']['content']
            # 提取分数和反馈
            score_match = re.search(r'(\d+)\/10', content)
            score = float(score_match.group(1))/10 if score_match else 0.5
            return {"score": score, "feedback": content}
        return {"score": 0.5, "feedback": "无法评估"}

    def tree_search_approach(self, problem: str, max_depth: int = 3) -> Dict:
        """改进的树搜索方法"""
        root = ThoughtNode(problem, "problem")
        subtasks = self._generate_subtasks(problem)
        
        for subtask in subtasks:
            subtask_node = ThoughtNode(subtask, "thought")
            root.children.append(subtask_node)
            
            # 对每个子任务生成解决方案
            messages = [
                {"role": "system", "content": "请解决以下子任务，并说明思考过程："},
                {"role": "user", "content": subtask}
            ]
            response = self._call_llm(messages)
            if 'choices' in response:
                solution = response['choices'][0]['message']['content']
                solution_node = ThoughtNode(solution, "action")
                
                # 评估解决方案
                evaluation = self._evaluate_solution(subtask, solution)
                solution_node.confidence = evaluation["score"]
                solution_node.feedback = evaluation["feedback"]
                
                subtask_node.children.append(solution_node)
                
                # 如果解决方案质量不够好，生成改进建议
                if evaluation["score"] < 0.7:
                    messages = [
                        {"role": "system", "content": "请根据以下反馈提出改进建议："},
                        {"role": "user", "content": evaluation["feedback"]}
                    ]
                    improvement = self._call_llm(messages)
                    if 'choices' in improvement:
                        reflection_node = ThoughtNode(
                            improvement['choices'][0]['message']['content'],
                            "reflection"
                        )
                        solution_node.children.append(reflection_node)

        return self._format_tree_response(root)

    def propose_critique_loop(self, problem: str, max_iterations: int = 3) -> Dict:
        """改进的提议-批评循环"""
        current_solution = None
        iterations = []
        
        for i in range(max_iterations):
            # 生成解决方案
            messages = [
                {"role": "system", "content": "请提出解决方案，考虑之前的反馈："},
                {"role": "user", "content": f"问题：{problem}\n当前解决方案：{current_solution if current_solution else '无'}"}
            ]
            proposal = self._call_llm(messages)
            if 'choices' in proposal:
                current_solution = proposal['choices'][0]['message']['content']
                
                # 评估解决方案
                evaluation = self._evaluate_solution(problem, current_solution)
                
                iterations.append({
                    "iteration": i + 1,
                    "proposal": current_solution,
                    "evaluation": evaluation
                })
                
                # 如果解决方案足够好，提前结束
                if evaluation["score"] >= 0.8:
                    break
        
        return self._format_loop_response(iterations)

    def multi_agent_debate(self, problem: str, num_agents: int = 3) -> Dict:
        """改进的多智能体辩论"""
        debate_history = []
        agents = [f"Agent_{i+1}" for i in range(num_agents)]
        
        # 初始化问题理解
        for agent in agents:
            messages = [
                {"role": "system", "content": f"你是{agent}，请从你的专业角度分析问题："},
                {"role": "user", "content": problem}
            ]
            response = self._call_llm(messages)
            if 'choices' in response:
                debate_history.append({
                    "agent": agent,
                    "content": response['choices'][0]['message']['content'],
                    "type": "analysis"
                })
        
        # 进行辩论
        for round in range(2):  # 可以调整轮数
            for agent in agents:
                # 让每个智能体评论其他智能体的观点
                messages = [
                    {"role": "system", "content": f"你是{agent}，请评论其他智能体的观点并提出自己的见解："},
                    {"role": "user", "content": str(debate_history)}
                ]
                response = self._call_llm(messages)
                if 'choices' in response:
                    debate_history.append({
                        "agent": agent,
                        "content": response['choices'][0]['message']['content'],
                        "type": "debate"
                    })
        
        # 总结辩论结果
        messages = [
            {"role": "system", "content": "请总结所有智能体的观点，提出最终解决方案："},
            {"role": "user", "content": str(debate_history)}
        ]
        conclusion = self._call_llm(messages)
        if 'choices' in conclusion:
            debate_history.append({
                "agent": "Moderator",
                "content": conclusion['choices'][0]['message']['content'],
                "type": "conclusion"
            })
        
        return self._format_debate_response(debate_history)

    def human_thought_annotation(self, problem: str) -> Dict:
        """改进的人类思维过程注释"""
        # 1. 问题分析
        messages = [
            {"role": "system", "content": "请像人类专家一样分析这个问题，包括关键点、难点和可能的解决方向："},
            {"role": "user", "content": problem}
        ]
        analysis = self._call_llm(messages)
        
        # 2. 解决方案探索
        messages = [
            {"role": "system", "content": "请探索多种可能的解决方案，并说明每种方案的优缺点："},
            {"role": "user", "content": problem}
        ]
        exploration = self._call_llm(messages)
        
        # 3. 方案改进
        if 'choices' in exploration:
            messages = [
                {"role": "system", "content": "请对提出的解决方案进行改进和优化："},
                {"role": "user", "content": exploration['choices'][0]['message']['content']}
            ]
            improvement = self._call_llm(messages)
        
        # 4. 最终总结
        messages = [
            {"role": "system", "content": "请总结整个思考过程，包括关键决策点和最终结论："},
            {"role": "user", "content": str([analysis, exploration, improvement])}
        ]
        conclusion = self._call_llm(messages)
        
        return self._format_annotation_response({
            "analysis": analysis['choices'][0]['message']['content'] if 'choices' in analysis else "",
            "exploration": exploration['choices'][0]['message']['content'] if 'choices' in exploration else "",
            "improvement": improvement['choices'][0]['message']['content'] if 'choices' in improvement else "",
            "conclusion": conclusion['choices'][0]['message']['content'] if 'choices' in conclusion else ""
        })

    def _format_tree_response(self, root: ThoughtNode) -> Dict:
        """格式化树搜索响应"""
        def node_to_dict(node: ThoughtNode) -> Dict:
            return {
                "content": node.content,
                "type": node.type,
                "confidence": node.confidence,
                "feedback": node.feedback,
                "children": [node_to_dict(child) for child in node.children]
            }
        return {"tree": node_to_dict(root)}

    def _format_loop_response(self, iterations: List[Dict]) -> Dict:
        """格式化循环响应"""
        return {"iterations": iterations}

    def _format_debate_response(self, debate_history: List[Dict]) -> Dict:
        """格式化辩论响应"""
        return {"debate_history": debate_history}

    def _format_annotation_response(self, annotation_data: Dict) -> Dict:
        """格式化注释响应"""
        return {"annotation": annotation_data}

    def generate_comprehensive_thought_chain(self, input_content: str) -> Dict:
        """生成综合思维链"""
        return {
            "input": input_content,
            "tree_search": self.tree_search_approach(input_content),
            "propose_critique": self.propose_critique_loop(input_content),
            "multi_agent": self.multi_agent_debate(input_content),
            "human_annotation": self.human_thought_annotation(input_content)
        }

def main():
    # Example usage
    generator = ThoughtChainGenerator()
    
    # Example problem (can be code or math problem)
    problem = """
    Optimize this code snippet:
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    thought_chain = generator.generate_comprehensive_thought_chain(problem)
    
    # Save results
    with open('thought_chain_output.json', 'w', encoding='utf-8') as f:
        json.dump(thought_chain, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
