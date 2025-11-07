"""
Comparative Benchmark: Traditional Tool Calling vs Code Mode

This benchmark runs the same queries through both approaches:
1. Traditional: LLM sees all tool schemas, makes tool call decisions
2. Code Mode: LLM writes TypeScript code to search/use tools

Metrics collected:
- Token usage (prompt + completion)
- Latency (total time)
- Success rate
- Tools found
- Number of LLM calls needed
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from code_mode_proper import CodeModeAgent
from gateway_dispatcher import ToolDatabase
from openai_integration import OpenAICodeGenerator
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results for a single query"""
    query: str
    approach: str  # "traditional" or "code_mode"
    success: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tools_found: int
    num_llm_calls: int
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraditionalToolCaller:
    """
    Traditional approach: LLM sees all tool schemas and makes tool calls
    
    This simulates the standard function calling approach where:
    1. All tool schemas are sent to LLM (EVERY REQUEST!)
    2. LLM returns function call
    3. We execute the function
    4. Results go back to LLM
    
    This demonstrates the MASSIVE token overhead of traditional approach.
    """
    
    def __init__(self, tool_database: ToolDatabase, api_key: Optional[str] = None):
        self.tool_database = tool_database
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # Use GPT-4-turbo (gpt-4-1106-preview) which has 128k context window
        # Regular GPT-4 only has 8k and will fail with many tools
        self.model = "gpt-4-1106-preview"
        
    def generate_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Generate OpenAI function schemas for ALL tools in the database.
        
        This is the REALISTIC traditional approach - send every tool as a separate function.
        With 47 tools, this creates 47 function schemas that all get sent to the LLM.
        """
        schemas = []
        
        # Create a function schema for EACH tool in the database
        for tool in self.tool_database.tools:
            schema = {
                "name": f"use_{tool['name'].lower().replace(' ', '_')}",
                "description": f"{tool.get('description', 'No description')} (Category: {tool.get('category', 'general')})",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform with this tool",
                            "enum": ["get_info", "execute", "validate"]
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for the tool"
                        }
                    },
                    "required": ["action"]
                }
            }
            schemas.append(schema)
        
        # Also add the generic search function
        schemas.append({
            "name": "search_tools",
            "description": "Search the tool database for tools matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding tools"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        })
        
        return schemas
        
    def process_query(self, user_query: str) -> BenchmarkResult:
        """Process query using traditional tool calling"""
        logger.info(f"[TRADITIONAL] Processing: {user_query}")
        
        start_time = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        num_llm_calls = 0
        tools_found = 0
        output = ""
        
        try:
            # Call 1: LLM decides what function to call
            # This sends ALL function schemas EVERY TIME!
            functions = self.generate_tool_schemas()
            
            logger.info(f"[TRADITIONAL] Sending {len(functions)} function schemas to LLM")
            
            response = self.client.chat.completions.create(
                model=self.model,  # Use GPT-4-turbo for larger context
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to a tool database. Use the available functions to help the user."
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                functions=functions,
                function_call="auto"
            )
            
            num_llm_calls += 1
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            total_tokens += response.usage.total_tokens
            
            logger.info(f"[TRADITIONAL] LLM Call 1: {response.usage.prompt_tokens} prompt tokens (includes ALL {len(functions)} function schemas!)")
            
            message = response.choices[0].message
            
            # Check if function was called
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                logger.info(f"[TRADITIONAL] LLM called: {function_name}({function_args})")
                
                # Execute the function
                if function_name == "search_tools":
                    query = function_args.get("query", "")
                    limit = function_args.get("limit", 5)
                    
                    results = self.tool_database.search(query, limit=limit)
                    tools_found = len(results)
                    
                    output = json.dumps({
                        "results": [
                            {
                                "name": r["name"],
                                "description": r.get("description", ""),
                                "category": r.get("category", ""),
                                "score": r.get("score", 0.0)
                            }
                            for r in results
                        ]
                    }, indent=2)
                    
                    # Call 2: Send function result back to LLM for final response
                    response2 = self.client.chat.completions.create(
                        model=self.model,  # Same model
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant with access to a tool database."
                            },
                            {
                                "role": "user",
                                "content": user_query
                            },
                            {
                                "role": "assistant",
                                "content": None,
                                "function_call": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args)
                                }
                            },
                            {
                                "role": "function",
                                "name": function_name,
                                "content": output
                            }
                        ]
                    )
                    
                    num_llm_calls += 1
                    prompt_tokens += response2.usage.prompt_tokens
                    completion_tokens += response2.usage.completion_tokens
                    total_tokens += response2.usage.total_tokens
                    
                    logger.info(f"[TRADITIONAL] LLM Call 2: {response2.usage.prompt_tokens} prompt tokens")
                    
                    final_response = response2.choices[0].message.content
                    logger.info(f"[TRADITIONAL] Final response: {final_response[:100]}...")
                    
            else:
                # LLM responded without function call
                output = message.content or "No function called"
                
            latency = (time.time() - start_time) * 1000
            
            return BenchmarkResult(
                query=user_query,
                approach="traditional",
                success=True,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tools_found=tools_found,
                num_llm_calls=num_llm_calls,
                output=output
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"[TRADITIONAL] Error: {e}")
            
            return BenchmarkResult(
                query=user_query,
                approach="traditional",
                success=False,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tools_found=0,
                num_llm_calls=num_llm_calls,
                output="",
                error=str(e)
            )


class CodeModeWrapper:
    """Wrapper for Code Mode to track metrics"""
    
    def __init__(self, agent: CodeModeAgent, llm_client: OpenAICodeGenerator):
        self.agent = agent
        self.llm_client = llm_client
        
    def process_query(self, user_query: str) -> BenchmarkResult:
        """Process query using Code Mode"""
        logger.info(f"[CODE MODE] Processing: {user_query}")
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self.agent.build_prompt(user_query)
            
            # Get code from LLM with metrics (returns a dict)
            result_dict = self.llm_client.generate_with_metrics(prompt)
            code = result_dict["code"]
            
            # Execute TypeScript
            result = self.agent.execute_typescript(code)
            
            latency = (time.time() - start_time) * 1000
            
            # Count tools found
            tools_found = 0
            if result.success and result.output:
                try:
                    output_data = json.loads(result.output)
                    tools_found = len(output_data.get("results", []))
                except:
                    pass
            
            return BenchmarkResult(
                query=user_query,
                approach="code_mode",
                success=result.success,
                latency_ms=latency,
                prompt_tokens=result_dict["prompt_tokens"],
                completion_tokens=result_dict["completion_tokens"],
                total_tokens=result_dict["tokens_used"],
                tools_found=tools_found,
                num_llm_calls=1,  # Code Mode only needs 1 LLM call
                output=result.output,
                error=result.error,
                metadata={
                    "code_generated": code,
                    "execution_time_ms": result.execution_time_ms,
                    "model": result_dict["model"]
                }
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"[CODE MODE] Error: {e}")
            
            return BenchmarkResult(
                query=user_query,
                approach="code_mode",
                success=False,
                latency_ms=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                tools_found=0,
                num_llm_calls=1,
                output="",
                error=str(e)
            )


def load_metatool_database(num_tools: int = 100) -> ToolDatabase:
    """
    Load the MetaTool database
    
    Args:
        num_tools: Number of tools to load (default: 100 for realistic benchmark)
    """
    data_path = Path(__file__).parent.parent.parent / "dataset" / "plugin_info.json"
    
    logger.info(f"Loading MetaTool database from: {data_path}")
    
    with open(data_path) as f:
        tools_data = json.load(f)
    
    tools = []
    
    # plugin_info.json is a list of tool objects - take first N tools
    if isinstance(tools_data, list):
        for tool_info in tools_data[:num_tools]:  # Limit to num_tools
            if isinstance(tool_info, dict):
                tools.append({
                    "name": tool_info.get("name_for_model", tool_info.get("name_for_human", "Unknown")),
                    "description": tool_info.get("description_for_model", tool_info.get("description_for_human", "")),
                    "category": "general",
                    "api_endpoint": f"https://api.example.com/{tool_info.get('name_for_model', 'unknown')}"
                })
    
    logger.info(f"✓ Loaded {len(tools)} tools (limited from {len(tools_data)} available)")
    return ToolDatabase(tools)


def get_test_queries() -> List[str]:
    """Get test queries"""
    return [
        "Find a tool to translate Spanish to English",
        "I need a weather forecasting service",
        "Search for image generation tools",
        "Find tools for sentiment analysis",
        "I want to convert currencies"
    ]


def run_comparative_benchmark():
    """Run comprehensive comparative benchmark"""
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK: Traditional vs Code Mode")
    print("=" * 80)
    print()
    
    # Load tool database with controlled size
    print("→ Loading MetaTool database...")
    tool_db = load_metatool_database(num_tools=100)  # Use 100 tools for realistic benchmark
    print()
    
    # Initialize both approaches
    print("→ Initializing Traditional Tool Caller...")
    traditional = TraditionalToolCaller(tool_db)
    
    # Show how many tool schemas are being sent
    schemas = traditional.generate_tool_schemas()
    schemas_json = json.dumps(schemas)
    schema_chars = len(schemas_json)
    # Rough token estimate: ~4 chars per token
    estimated_schema_tokens = schema_chars // 4
    
    print(f"✓ Traditional approach ready")
    print(f"  → Sending {len(schemas)} function schemas to LLM")
    print(f"  → Schema size: {schema_chars:,} characters (~{estimated_schema_tokens:,} tokens)")
    print()
    
    print("→ Initializing Code Mode...")
    llm_client = OpenAICodeGenerator(model="gpt-4")
    code_mode_agent = CodeModeAgent(
        tool_database=tool_db,
        llm_client=llm_client,
        proxy_port=3002  # Different port to avoid conflicts
    )
    
    if not code_mode_agent.deno_available:
        print("ERROR: Deno not available!")
        return
        
    code_mode = CodeModeWrapper(code_mode_agent, llm_client)
    print("✓ Code Mode ready")
    print()
    
    # Get test queries
    test_queries = get_test_queries()
    
    print("=" * 80)
    print(f"RUNNING {len(test_queries)} QUERIES WITH BOTH APPROACHES")
    print("=" * 80)
    print()
    
    all_results = []
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(test_queries)}: {query}")
            print(f"{'='*80}\n")
            
            # Run Traditional approach
            print("→ Running Traditional Tool Calling...")
            trad_result = traditional.process_query(query)
            all_results.append(trad_result)
            
            print(f"  ✓ Success: {trad_result.success}")
            print(f"  ✓ Latency: {trad_result.latency_ms:.0f}ms")
            print(f"  ✓ Tokens: {trad_result.total_tokens} (prompt: {trad_result.prompt_tokens}, completion: {trad_result.completion_tokens})")
            print(f"  ✓ LLM calls: {trad_result.num_llm_calls}")
            print(f"  ✓ Tools found: {trad_result.tools_found}")
            print()
            
            time.sleep(1)  # Rate limiting
            
            # Run Code Mode approach
            print("→ Running Code Mode...")
            cm_result = code_mode.process_query(query)
            all_results.append(cm_result)
            
            print(f"  ✓ Success: {cm_result.success}")
            print(f"  ✓ Latency: {cm_result.latency_ms:.0f}ms")
            print(f"  ✓ Tokens: {cm_result.total_tokens} (prompt: {cm_result.prompt_tokens}, completion: {cm_result.completion_tokens})")
            print(f"  ✓ LLM calls: {cm_result.num_llm_calls}")
            print(f"  ✓ Tools found: {cm_result.tools_found}")
            print()
            
            # Show comparison
            token_reduction = ((trad_result.total_tokens - cm_result.total_tokens) / trad_result.total_tokens * 100) if trad_result.total_tokens > 0 else 0
            latency_diff = cm_result.latency_ms - trad_result.latency_ms
            
            print(f"📊 COMPARISON:")
            print(f"  Token reduction: {token_reduction:+.1f}%")
            print(f"  Latency difference: {latency_diff:+.0f}ms")
            print()
            
            time.sleep(1)  # Rate limiting
            
    finally:
        code_mode_agent.shutdown()
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS")
    print("=" * 80)
    print()
    
    # Separate results by approach
    trad_results = [r for r in all_results if r.approach == "traditional"]
    cm_results = [r for r in all_results if r.approach == "code_mode"]
    
    # Calculate aggregates
    trad_avg_tokens = sum(r.total_tokens for r in trad_results) / len(trad_results) if trad_results else 0
    cm_avg_tokens = sum(r.total_tokens for r in cm_results) / len(cm_results) if cm_results else 0
    
    trad_avg_latency = sum(r.latency_ms for r in trad_results) / len(trad_results) if trad_results else 0
    cm_avg_latency = sum(r.latency_ms for r in cm_results) / len(cm_results) if cm_results else 0
    
    trad_success_rate = sum(1 for r in trad_results if r.success) / len(trad_results) * 100 if trad_results else 0
    cm_success_rate = sum(1 for r in cm_results if r.success) / len(cm_results) * 100 if cm_results else 0
    
    trad_avg_llm_calls = sum(r.num_llm_calls for r in trad_results) / len(trad_results) if trad_results else 0
    cm_avg_llm_calls = sum(r.num_llm_calls for r in cm_results) / len(cm_results) if cm_results else 0
    
    token_reduction_pct = ((trad_avg_tokens - cm_avg_tokens) / trad_avg_tokens * 100) if trad_avg_tokens > 0 else 0
    latency_reduction_pct = ((trad_avg_latency - cm_avg_latency) / trad_avg_latency * 100) if trad_avg_latency > 0 else 0
    
    print("TRADITIONAL TOOL CALLING:")
    print(f"  Average Tokens: {trad_avg_tokens:.0f}")
    print(f"  Average Latency: {trad_avg_latency:.0f}ms")
    print(f"  Success Rate: {trad_success_rate:.1f}%")
    print(f"  Average LLM Calls: {trad_avg_llm_calls:.1f}")
    print()
    
    print("CODE MODE:")
    print(f"  Average Tokens: {cm_avg_tokens:.0f}")
    print(f"  Average Latency: {cm_avg_latency:.0f}ms")
    print(f"  Success Rate: {cm_success_rate:.1f}%")
    print(f"  Average LLM Calls: {cm_avg_llm_calls:.1f}")
    print()
    
    print("IMPROVEMENTS:")
    print(f"  Token Reduction: {token_reduction_pct:+.1f}%")
    print(f"  Latency Change: {latency_reduction_pct:+.1f}%")
    print(f"  LLM Call Reduction: {trad_avg_llm_calls - cm_avg_llm_calls:+.1f}")
    print()
    
    # Save results
    output_dir = Path(__file__).parent / "results" / "comparative_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed JSON results
    results_file = output_dir / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "traditional": {
                    "avg_tokens": trad_avg_tokens,
                    "avg_latency_ms": trad_avg_latency,
                    "success_rate": trad_success_rate,
                    "avg_llm_calls": trad_avg_llm_calls
                },
                "code_mode": {
                    "avg_tokens": cm_avg_tokens,
                    "avg_latency_ms": cm_avg_latency,
                    "success_rate": cm_success_rate,
                    "avg_llm_calls": cm_avg_llm_calls
                },
                "improvements": {
                    "token_reduction_pct": token_reduction_pct,
                    "latency_reduction_pct": latency_reduction_pct,
                    "llm_call_reduction": trad_avg_llm_calls - cm_avg_llm_calls
                }
            },
            "detailed_results": [
                {
                    "query": r.query,
                    "approach": r.approach,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "tools_found": r.tools_found,
                    "num_llm_calls": r.num_llm_calls,
                    "error": r.error
                }
                for r in all_results
            ]
        }, f, indent=2)
    
    print(f"✓ Detailed results: {results_file}")
    
    # Human-readable summary
    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("COMPARATIVE BENCHMARK: Traditional vs Code Mode\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Queries: {len(test_queries)}\n")
        f.write(f"Tool Database Size: {len(tool_db.tools)} tools\n\n")
        
        f.write("TRADITIONAL TOOL CALLING:\n")
        f.write(f"  Average Tokens: {trad_avg_tokens:.0f}\n")
        f.write(f"  Average Latency: {trad_avg_latency:.0f}ms\n")
        f.write(f"  Success Rate: {trad_success_rate:.1f}%\n")
        f.write(f"  Average LLM Calls: {trad_avg_llm_calls:.1f}\n\n")
        
        f.write("CODE MODE:\n")
        f.write(f"  Average Tokens: {cm_avg_tokens:.0f}\n")
        f.write(f"  Average Latency: {cm_avg_latency:.0f}ms\n")
        f.write(f"  Success Rate: {cm_success_rate:.1f}%\n")
        f.write(f"  Average LLM Calls: {cm_avg_llm_calls:.1f}\n\n")
        
        f.write("IMPROVEMENTS:\n")
        f.write(f"  Token Reduction: {token_reduction_pct:+.1f}%\n")
        f.write(f"  Latency Change: {latency_reduction_pct:+.1f}%\n")
        f.write(f"  LLM Call Reduction: {trad_avg_llm_calls - cm_avg_llm_calls:+.1f}\n\n")
        
        f.write("\nPER-QUERY BREAKDOWN:\n")
        f.write("-" * 80 + "\n")
        
        for i, query in enumerate(test_queries):
            trad = trad_results[i]
            cm = cm_results[i]
            
            token_red = ((trad.total_tokens - cm.total_tokens) / trad.total_tokens * 100) if trad.total_tokens > 0 else 0
            
            f.write(f"\n{i+1}. {query}\n")
            f.write(f"   Traditional: {trad.total_tokens} tokens, {trad.latency_ms:.0f}ms, {trad.num_llm_calls} LLM calls\n")
            f.write(f"   Code Mode:   {cm.total_tokens} tokens, {cm.latency_ms:.0f}ms, {cm.num_llm_calls} LLM calls\n")
            f.write(f"   Improvement: {token_red:+.1f}% tokens\n")
    
    print(f"✓ Summary report: {summary_file}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_comparative_benchmark()
