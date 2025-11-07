"""
Gateway Meta-Tool Dispatcher

Core component that compresses thousands of tool schemas into a single meta-tool
and generates executable code for tool retrieval and validation.

This is the key innovation: instead of sending 10K tool schemas to the LLM,
we send 3 meta-tool functions and let the LLM generate code to search/validate.
"""

import json
import time
import tiktoken
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import re
import ast


@dataclass
class MetaToolResult:
    """Result from meta-tool execution."""
    success: bool
    result: Any
    execution_time_ms: float
    tokens_used: int
    code_generated: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolDatabase:
    """Simulates a vector database for tool retrieval."""
    
    def __init__(self, tools: List[Dict[str, Any]]):
        """Initialize with tool library."""
        self.tools = tools
        self.name_index = {tool["name"]: tool for tool in tools}
        self.category_index = self._build_category_index()
    
    def _build_category_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index by category for faster filtering."""
        index = {}
        for tool in self.tools:
            category = tool.get("category", "other")
            if category not in index:
                index[category] = []
            index[category].append(tool)
        return index
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search simulation.
        In production, this would use vector embeddings and cosine similarity.
        For MVP, we use keyword matching.
        """
        query_lower = query.lower()
        results = []
        
        for tool in self.tools:
            score = 0.0
            
            # Check name match
            if query_lower in tool["name"].lower():
                score += 0.5
            
            # Check description match
            desc = tool.get("description", "").lower()
            query_words = query_lower.split()
            for word in query_words:
                if word in desc:
                    score += 0.1
            
            # Check category match
            if filters and "category" in filters:
                filter_cats = filters["category"] if isinstance(filters["category"], list) else [filters["category"]]
                if tool.get("category") in filter_cats:
                    score += 0.3
            
            if score > 0:
                result = tool.copy()
                result["score"] = round(score, 3)
                results.append(result)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool by exact name."""
        return self.name_index.get(tool_name)
    
    def get_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all tools in a category."""
        return self.category_index.get(category, [])[:limit]
    
    def validate_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against tool schema."""
        tool = self.get_by_name(tool_name)
        if not tool:
            return {"valid": False, "error": f"Tool '{tool_name}' not found"}
        
        tool_params = tool.get("parameters", {})
        errors = []
        
        # Check required parameters
        for param_name, param_def in tool_params.items():
            if param_def.get("required", False) and param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
        
        # Check parameter types (simplified)
        for param_name, param_value in params.items():
            if param_name in tool_params:
                expected_type = tool_params[param_name].get("type")
                # Type checking would be more robust in production
                if expected_type == "integer" and not isinstance(param_value, int):
                    errors.append(f"Parameter '{param_name}' should be integer")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "tool": tool}


class GatewayMetaToolDispatcher:
    """
    Main gateway component that:
    1. Intercepts user requests
    2. Provides compressed meta-tool schema to LLM
    3. Executes generated code
    4. Returns results
    """
    
    # Meta-tool schema (this is what goes to the LLM instead of 10K tools)
    META_TOOL_SCHEMA = {
        "meta_tool_search": {
            "name": "meta_tool_search",
            "description": "Search the tool database using semantic search. Returns top matching tools.",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the desired tool functionality",
                    "required": True
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters (e.g., {'category': ['weather', 'travel']})",
                    "required": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "required": False,
                    "default": 5
                }
            },
            "returns": "List of tool objects with name, description, category, and relevance score"
        },
        "meta_tool_get_by_name": {
            "name": "meta_tool_get_by_name",
            "description": "Retrieve a specific tool by its exact name",
            "parameters": {
                "tool_name": {
                    "type": "string",
                    "description": "The exact name of the tool to retrieve",
                    "required": True
                }
            },
            "returns": "Tool object or None if not found"
        },
        "meta_tool_validate_params": {
            "name": "meta_tool_validate_params",
            "description": "Validate parameters against a tool's schema",
            "parameters": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to validate against",
                    "required": True
                },
                "params": {
                    "type": "object",
                    "description": "Parameters to validate",
                    "required": True
                }
            },
            "returns": "Validation result with 'valid' boolean and optional 'errors' list"
        },
        "meta_tool_get_by_category": {
            "name": "meta_tool_get_by_category",
            "description": "Get all tools in a specific category",
            "parameters": {
                "category": {
                    "type": "string",
                    "description": "Category name (e.g., 'weather', 'translation', 'finance')",
                    "required": True
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "required": False,
                    "default": 10
                }
            },
            "returns": "List of tools in the category"
        }
    }
    
    def __init__(self, tool_database: ToolDatabase, llm_client: Optional[Callable] = None):
        """
        Initialize gateway dispatcher.
        
        Args:
            tool_database: Database of available tools
            llm_client: Function to call LLM (takes prompt, returns code)
        """
        self.tool_db = tool_database
        self.llm_client = llm_client or self._mock_llm_client
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    def _mock_llm_client(self, prompt: str) -> str:
        """
        Mock LLM client for testing.
        In production, this would call OpenAI/Claude/etc.
        """
        # Simple pattern matching for demo purposes
        if "air quality" in prompt.lower():
            return """
results = meta_tool_search(
    query="air quality forecast zip code",
    filters={"category": ["weather", "environment"]},
    limit=3
)

best_tool = results[0]
return best_tool
"""
        elif "translate" in prompt.lower():
            return """
results = meta_tool_search(
    query="translate language translation",
    filters={"category": ["translation"]},
    limit=5
)

for tool in results:
    if "spanish" in tool["description"].lower() or "language" in tool["description"].lower():
        validation = meta_tool_validate_params(
            tool_name=tool["name"],
            params={"source": "es", "target": "en"}
        )
        if validation.get("valid"):
            return tool

return results[0]
"""
        else:
            return f"""
results = meta_tool_search(
    query="{prompt[:100]}",
    limit=5
)
return results[0] if results else None
"""
    
    def build_compressed_prompt(self, user_query: str) -> str:
        """
        Build the compressed prompt with meta-tool schema.
        This is sent to the LLM instead of 10K tool schemas.
        """
        meta_tools_desc = json.dumps(self.META_TOOL_SCHEMA, indent=2)
        
        prompt = f"""You are an intelligent tool retrieval system. Your task is to write Python code that finds the best tool for the user's request.

User Query: "{user_query}"

Available Functions:
You have access to these functions to search and validate tools in a database of 10,000+ tools:

{meta_tools_desc}

Tool Database Info:
- Total tools: {len(self.tool_db.tools)}
- Categories: {list(self.tool_db.category_index.keys())}
- Supports semantic search: Yes
- Supports parameter validation: Yes

Instructions:
1. Write Python code to find the most appropriate tool(s) for the user query
2. Use meta_tool_search() to find relevant tools
3. Use meta_tool_validate_params() to check parameter compatibility if needed
4. Return the best matching tool(s)
5. Your code should be efficient and handle edge cases

Generate only the Python code (no explanations):
"""
        return prompt
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def execute_code_safely(self, code: str) -> Any:
        """
        Execute generated code in a controlled environment.
        In production, this would use a proper sandbox.
        """
        # Create safe execution environment
        safe_globals = {
            "meta_tool_search": self.tool_db.search,
            "meta_tool_get_by_name": self.tool_db.get_by_name,
            "meta_tool_validate_params": self.tool_db.validate_params,
            "meta_tool_get_by_category": self.tool_db.get_by_category,
            "__builtins__": {
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
            }
        }
        
        try:
            # Execute code
            exec(code, safe_globals)
            
            # Get return value (last expression)
            # In production, we'd parse the AST properly
            if "return" in code:
                # Re-execute to capture return value
                code_lines = code.strip().split('\n')
                for i, line in enumerate(code_lines):
                    if line.strip().startswith('return'):
                        result_code = '\n'.join(code_lines[:i+1])
                        # Wrap in function to capture return
                        func_code = f"def _exec_func():\n" + '\n'.join('    ' + l for l in result_code.split('\n'))
                        exec(func_code, safe_globals)
                        return safe_globals['_exec_func']()
            
            # If no explicit return, try to get the last assigned variable
            return None
            
        except Exception as e:
            raise RuntimeError(f"Code execution error: {str(e)}")
    
    def dispatch(self, user_query: str) -> MetaToolResult:
        """
        Main dispatch method: Gateway intercepts request and orchestrates retrieval.
        
        This is where the magic happens:
        1. Build compressed prompt (2K tokens)
        2. LLM generates retrieval code
        3. Execute code against tool DB
        4. Return result
        """
        start_time = time.time()
        
        # Step 1: Build compressed prompt
        compressed_prompt = self.build_compressed_prompt(user_query)
        prompt_tokens = self.count_tokens(compressed_prompt)
        
        # Step 2: Call LLM to generate code
        try:
            generated_code = self.llm_client(compressed_prompt)
            code_tokens = self.count_tokens(generated_code)
            total_tokens = prompt_tokens + code_tokens
            
            # Step 3: Execute code
            result = self.execute_code_safely(generated_code)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            return MetaToolResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                tokens_used=total_tokens,
                code_generated=generated_code,
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "code_tokens": code_tokens,
                    "user_query": user_query
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return MetaToolResult(
                success=False,
                result=None,
                execution_time_ms=execution_time,
                tokens_used=prompt_tokens,
                code_generated="",
                error=str(e),
                metadata={"user_query": user_query}
            )


def demo():
    """Demonstrate the gateway meta-tool dispatcher."""
    print("="*70)
    print("GATEWAY META-TOOL DISPATCHER DEMO")
    print("="*70)
    
    # Load synthetic tools (or use sample)
    sample_tools = [
        {
            "id": 1,
            "name": "airqualityforeast",
            "category": "weather",
            "description": "Planning something outdoors? Get the 2-day air quality forecast for any US zip code.",
            "parameters": {"zip_code": {"type": "string", "required": True}}
        },
        {
            "id": 2,
            "name": "MixerBox_Translate_AI_language_tutor",
            "category": "translation",
            "description": "Translate any language right away! Learn foreign languages easily by conversing with AI tutors!",
            "parameters": {"text": {"type": "string", "required": True}, "target_lang": {"type": "string", "required": True}}
        },
        {
            "id": 3,
            "name": "calculator",
            "category": "productivity",
            "description": "A calculator app that executes a given formula and returns a result. This app can execute basic and advanced operations.",
            "parameters": {"formula": {"type": "string", "required": True}}
        }
    ]
    
    # Initialize
    tool_db = ToolDatabase(sample_tools)
    dispatcher = GatewayMetaToolDispatcher(tool_db)
    
    # Test queries
    test_queries = [
        "I need to check air quality in New York, zip 10001",
        "Help me translate Spanish to English",
        "Calculate 2 + 2"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        result = dispatcher.dispatch(query)
        
        if result.success:
            print(f"\n✓ SUCCESS")
            print(f"  Execution Time: {result.execution_time_ms:.2f}ms")
            print(f"  Tokens Used: {result.tokens_used:,}")
            print(f"\n  Generated Code:")
            print("  " + "\n  ".join(result.code_generated.split('\n')))
            print(f"\n  Result:")
            if isinstance(result.result, dict):
                print(f"    Tool: {result.result.get('name', 'N/A')}")
                print(f"    Category: {result.result.get('category', 'N/A')}")
                print(f"    Score: {result.result.get('score', 'N/A')}")
            else:
                print(f"    {result.result}")
        else:
            print(f"\n✗ FAILED: {result.error}")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
