import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class ThoughtChainGenerator:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _call_llm(self, messages):
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "stream": False
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        return response.json()

    def tree_search_approach(self, problem, max_steps=5):
        """Attempt 1: Tree Search with LLM and Reward"""
        messages = [
            {"role": "system", "content": "You are a problem solver that explores multiple paths and learns from mistakes."},
            {"role": "user", "content": f"""
            Solve this problem step by step. For each step:
            1. Propose a solution step
            2. Evaluate if it's correct
            3. If incorrect, backtrack and try a different approach
            4. Document your thought process
            
            Problem: {problem}
            """}
        ]
        return self._call_llm(messages)

    def propose_critique_loop(self, problem):
        """Attempt 2: Propose-Critique Loop"""
        messages = [
            {"role": "system", "content": "You are a dual-role problem solver: proposer and critic."},
            {"role": "user", "content": f"""
            Solve this problem using the following format:
            1. PROPOSE: Suggest a solution step
            2. CRITIQUE: Analyze the proposal
            3. ACTION: Choose to continue, backtrack, reflect, or terminate
            
            Problem: {problem}
            """}
        ]
        return self._call_llm(messages)

    def multi_agent_debate(self, problem):
        """Attempt 3: Multi-Agent Approach"""
        messages = [
            {"role": "system", "content": "You are two agents: a solver and a critic, working together to solve a problem."},
            {"role": "user", "content": f"""
            Engage in a debate to solve this problem:
            SOLVER: Propose solution steps
            CRITIC: Evaluate steps and suggest improvements
            Both: Document the reasoning process
            
            Problem: {problem}
            """}
        ]
        return self._call_llm(messages)

    def human_thought_annotation(self, problem):
        """Attempt 4: Human Thought Process Annotation"""
        messages = [
            {"role": "system", "content": "You are an expert at modeling human-like problem-solving processes."},
            {"role": "user", "content": f"""
            Solve this problem while documenting your thought process like a human would:
            1. Initial thoughts
            2. Attempts and reflections
            3. Mistakes and corrections
            4. Final solution path
            
            Problem: {problem}
            """}
        ]
        return self._call_llm(messages)

    def generate_comprehensive_thought_chain(self, input_content):
        """Combines all approaches to generate a comprehensive thought chain"""
        thought_chain = {
            "input": input_content,
            "tree_search": self.tree_search_approach(input_content),
            "propose_critique": self.propose_critique_loop(input_content),
            "multi_agent": self.multi_agent_debate(input_content),
            "human_annotation": self.human_thought_annotation(input_content)
        }
        
        return thought_chain

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
