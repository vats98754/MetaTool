"""
Baseline Meta-Tool Evaluator

Implements the traditional Meta-Tool approach where the LLM receives ALL tool schemas
and must select the appropriate tool. This serves as the baseline for comparison.

Token usage: ~150K tokens (sending all tool schemas to LLM)
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class BaselineResult:
    """Result from baseline Meta-Tool evaluation."""
    success: bool
    selected_tool: Optional[Dict[str, Any]]
    execution_time_ms: float
    tokens_used: int
    prompt_sent: str
    llm_response: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaselineMetaToolEvaluator:
    """
    Baseline Meta-Tool approach:
    1. Receives user query
    2. Sends ALL tool schemas to LLM (massive prompt)
    3. LLM selects best tool from the list
    4. Returns selected tool
    """
    
    def __init__(self, tools: List[Dict[str, Any]], llm_client: Optional[callable] = None):
        """
        Initialize baseline evaluator.
        
        Args:
            tools: Full list of tool schemas
            llm_client: Function to call LLM (takes prompt, returns response)
        """
        self.tools = tools
        self.llm_client = llm_client or self._mock_llm_baseline_client
    
    def _mock_llm_baseline_client(self, prompt: str) -> str:
        """
        Mock LLM client for baseline approach.
        Simulates LLM selecting a tool from the massive list.
        """
        # Simple pattern matching for demo
        prompt_lower = prompt.lower()
        
        if "air quality" in prompt_lower:
            return "airqualityforeast"
        elif "translate" in prompt_lower or "spanish" in prompt_lower:
            return "MixerBox_Translate_AI_language_tutor"
        elif "calculate" in prompt_lower or "math" in prompt_lower:
            return "calculator"
        elif "weather" in prompt_lower:
            # Find first weather tool
            for tool in self.tools:
                if tool.get("category") == "weather":
                    return tool["name"]
        
        # Default: return first tool
        return self.tools[0]["name"] if self.tools else "None"
    
    def build_baseline_prompt(self, user_query: str) -> str:
        """
        Build the traditional prompt with ALL tool schemas.
        This is the massive prompt that causes 150K token usage.
        """
        # Format all tools for the prompt
        tool_list = []
        for i, tool in enumerate(self.tools, 1):
            tool_entry = f"{i}. {tool['name']}: {tool.get('description', 'No description')}"
            
            # Add parameters if available (increases token count significantly)
            params = tool.get('parameters', {})
            if params:
                param_strs = []
                for param_name, param_def in params.items():
                    param_type = param_def.get('type', 'any')
                    required = ' (required)' if param_def.get('required') else ' (optional)'
                    param_strs.append(f"    - {param_name}: {param_type}{required}")
                if param_strs:
                    tool_entry += "\n" + "\n".join(param_strs)
            
            tool_list.append(tool_entry)
        
        tools_text = "\n\n".join(tool_list)
        
        prompt = f"""You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user's query based on their question.

I will provide you with the user's question and information about ALL available tools in the system.

User's Query:
[User's Query Start]
{user_query}
[User's Query End]

List of ALL Available Tools with Names and Descriptions:
[List of Tools with Names and Descriptions Start]
{tools_text}
[List of Tools with Names and Descriptions End]

Instructions:
1. Carefully read the user's query
2. Review ALL tools in the list above
3. Select the BEST tool that matches the user's needs
4. If there is a tool that is applicable, return ONLY the tool name
5. If there isn't a good match, return 'None'

Respond with ONLY the tool name (nothing else):
"""
        return prompt
    
    def count_tokens_estimate(self, text: str) -> int:
        """
        Estimate token count.
        Using simple heuristic: ~1 token per 4 characters for English text.
        This is conservative; actual tokens are usually more.
        """
        # More accurate estimate for technical text with JSON-like structure
        # Average ~3.5 characters per token for technical content
        return int(len(text) / 3.5)
    
    def evaluate(self, user_query: str) -> BaselineResult:
        """
        Evaluate using baseline Meta-Tool approach.
        
        This sends the FULL tool list to the LLM (massive token usage).
        """
        start_time = time.time()
        
        try:
            # Step 1: Build massive prompt with all tools
            prompt = self.build_baseline_prompt(user_query)
            prompt_tokens = self.count_tokens_estimate(prompt)
            
            # Step 2: Call LLM with massive prompt
            llm_response = self.llm_client(prompt)
            response_tokens = self.count_tokens_estimate(llm_response)
            total_tokens = prompt_tokens + response_tokens
            
            # Step 3: Parse response to get tool name
            tool_name = llm_response.strip()
            
            # Step 4: Find the selected tool
            selected_tool = None
            for tool in self.tools:
                if tool["name"] == tool_name:
                    selected_tool = tool
                    break
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            return BaselineResult(
                success=selected_tool is not None,
                selected_tool=selected_tool,
                execution_time_ms=execution_time,
                tokens_used=total_tokens,
                prompt_sent=prompt[:500] + "..." if len(prompt) > 500 else prompt,  # Truncate for storage
                llm_response=llm_response,
                metadata={
                    "user_query": user_query,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "tools_in_prompt": len(self.tools),
                    "avg_tokens_per_tool": prompt_tokens / len(self.tools) if self.tools else 0
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return BaselineResult(
                success=False,
                selected_tool=None,
                execution_time_ms=execution_time,
                tokens_used=0,
                prompt_sent="",
                llm_response="",
                error=str(e),
                metadata={"user_query": user_query}
            )


def demo():
    """Demonstrate the baseline Meta-Tool evaluator."""
    print("="*70)
    print("BASELINE META-TOOL EVALUATOR DEMO")
    print("="*70)
    
    # Create sample tools (in reality, this would be 10K+ tools)
    sample_tools = [
        {
            "id": 1,
            "name": "airqualityforeast",
            "category": "weather",
            "description": "Planning something outdoors? Get the 2-day air quality forecast for any US zip code.",
            "parameters": {
                "zip_code": {"type": "string", "required": True, "description": "US zip code"}
            }
        },
        {
            "id": 2,
            "name": "MixerBox_Translate_AI_language_tutor",
            "category": "translation",
            "description": "Translate any language right away! Learn foreign languages easily by conversing with AI tutors!",
            "parameters": {
                "text": {"type": "string", "required": True, "description": "Text to translate"},
                "source_lang": {"type": "string", "required": False, "description": "Source language code"},
                "target_lang": {"type": "string", "required": True, "description": "Target language code"}
            }
        },
        {
            "id": 3,
            "name": "calculator",
            "category": "productivity",
            "description": "A calculator app that executes a given formula and returns a result. This app can execute basic and advanced operations.",
            "parameters": {
                "formula": {"type": "string", "required": True, "description": "Mathematical formula to calculate"}
            }
        },
        {
            "id": 4,
            "name": "WeatherForecast",
            "category": "weather",
            "description": "Get detailed weather forecasts for any location worldwide. Includes temperature, precipitation, wind, and more.",
            "parameters": {
                "location": {"type": "string", "required": True, "description": "City name or coordinates"},
                "days": {"type": "integer", "required": False, "description": "Number of days to forecast"}
            }
        }
    ]
    
    # Simulate having many more tools (multiply the sample)
    # In real test, we'd load 10K tools
    simulated_large_tool_list = sample_tools * 250  # 1000 tools
    for i, tool in enumerate(simulated_large_tool_list):
        tool = tool.copy()
        tool["id"] = i + 1
        tool["name"] = f"{tool['name']}_{i % 250}" if i >= 4 else tool["name"]
        simulated_large_tool_list[i] = tool
    
    print(f"\nUsing {len(simulated_large_tool_list)} tools for simulation")
    
    # Initialize evaluator
    evaluator = BaselineMetaToolEvaluator(simulated_large_tool_list)
    
    # Test queries
    test_queries = [
        "I need to check air quality in New York, zip 10001",
        "Help me translate Spanish to English",
        "What's the weather like in San Francisco?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        result = evaluator.evaluate(query)
        
        if result.success:
            print(f"\n✓ SUCCESS")
            print(f"  Execution Time: {result.execution_time_ms:.2f}ms")
            print(f"  Tokens Used: {result.tokens_used:,}")
            print(f"  Tokens in Prompt: {result.metadata['prompt_tokens']:,}")
            print(f"  Tools in Prompt: {result.metadata['tools_in_prompt']:,}")
            print(f"  Avg Tokens per Tool: {result.metadata['avg_tokens_per_tool']:.1f}")
            print(f"\n  Selected Tool:")
            print(f"    Name: {result.selected_tool['name']}")
            print(f"    Category: {result.selected_tool.get('category', 'N/A')}")
            print(f"    Description: {result.selected_tool.get('description', 'N/A')[:100]}...")
        else:
            print(f"\n✗ FAILED: {result.error}")
    
    print(f"\n{'='*70}")
    print("BASELINE DEMO COMPLETE")
    print(f"{'='*70}")
    
    # Print comparison insight
    print("\n📊 KEY INSIGHT:")
    print(f"  With {len(simulated_large_tool_list)} tools:")
    print(f"  - Each query uses ~{result.metadata['prompt_tokens']:,} tokens")
    print(f"  - At $0.01/1K tokens: ${result.metadata['prompt_tokens'] * 0.01 / 1000:.4f} per query")
    print(f"  - For 1,000 queries: ${result.metadata['prompt_tokens'] * 0.01 / 1000 * 1000:.2f}")
    print(f"\n  With 10,000 tools, token usage would be ~10x higher!")


if __name__ == "__main__":
    demo()
