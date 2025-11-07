"""
Benchmark for Proper Code Mode Implementation (with Deno + OpenAI)

This uses the CORRECT Code Mode architecture:
- TypeScript code generation (not Python)
- Deno execution (not subprocess Python)
- HTTP proxy for tool access
- Real fetch() calls from TypeScript

Based on: jx-codes/lootbox and Cloudflare Code Mode
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from code_mode_proper import CodeModeAgent
from gateway_dispatcher import ToolDatabase
from openai_integration import OpenAICodeGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_metatool_database() -> ToolDatabase:
    """Load the full MetaTool database"""
    data_path = Path(__file__).parent.parent.parent / "dataset" / "big_tool_des.json"
    
    logger.info(f"Loading MetaTool database from: {data_path}")
    
    with open(data_path) as f:
        tools_data = json.load(f)
    
    # Convert to expected format - values are description strings
    tools = []
    for tool_name, description in tools_data.items():
        tools.append({
            "name": tool_name,
            "description": description if isinstance(description, str) else "",
            "category": "general",  # Could parse from description or tool name
            "api_endpoint": f"https://api.example.com/{tool_name.lower()}"
        })
    
    logger.info(f"✓ Loaded {len(tools)} tools")
    return ToolDatabase(tools)


def get_test_queries() -> List[Dict[str, Any]]:
    """Get test queries for benchmarking"""
    return [
        {
            "query": "Find a tool to translate Spanish to English",
            "expected_category": "translation",
            "description": "Simple translation tool search"
        },
        {
            "query": "I need a weather forecasting service",
            "expected_category": "weather",
            "description": "Weather API search"
        },
        {
            "query": "Search for image generation tools",
            "expected_category": "image",
            "description": "Image generation search"
        },
        {
            "query": "Find tools for sentiment analysis",
            "expected_category": "nlp",
            "description": "NLP/sentiment analysis search"
        },
        {
            "query": "I want to convert currencies",
            "expected_category": "finance",
            "description": "Currency conversion search"
        }
    ]


def run_benchmark():
    """Run comprehensive benchmark of proper Code Mode"""
    print("\n" + "=" * 80)
    print("PROPER CODE MODE BENCHMARK (Deno + OpenAI)")
    print("=" * 80)
    print()
    
    # Load tool database
    print("→ Loading MetaTool database...")
    tool_db = load_metatool_database()
    print()
    
    # Initialize OpenAI client
    print("→ Initializing OpenAI GPT-4...")
    try:
        llm_client = OpenAICodeGenerator(model="gpt-4")
        print("✓ OpenAI client ready")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI: {e}")
        print("\nMake sure OPENAI_API_KEY is set in .env file")
        return
    print()
    
    # Create Code Mode agent
    print("→ Creating Code Mode agent...")
    agent = CodeModeAgent(
        tool_database=tool_db,
        llm_client=llm_client,
        proxy_port=3001
    )
    
    if not agent.deno_available:
        print("\n" + "!" * 80)
        print("ERROR: Deno is not installed!")
        print("!" * 80)
        print("\nInstall: curl -fsSL https://deno.land/install.sh | sh")
        return
        
    print("✓ Agent ready with Deno runtime")
    print()
    
    # Get test queries
    test_queries = get_test_queries()
    
    print("=" * 80)
    print(f"RUNNING {len(test_queries)} QUERIES")
    print("=" * 80)
    print()
    
    results = []
    
    try:
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(test_queries)}: {test_case['query']}")
            print(f"{'='*80}")
            
            result = agent.process_query(test_case['query'])
            
            # Add test case info
            result['test_case'] = test_case
            result['query_number'] = i
            
            results.append(result)
            
            # Show summary
            print(f"\n→ Success: {result['success']}")
            print(f"→ Time: {result['total_time_ms']:.0f}ms")
            if result['success'] and result['output']:
                try:
                    output_data = json.loads(result['output'])
                    num_results = len(output_data.get('results', []))
                    print(f"→ Found {num_results} tools")
                except:
                    print(f"→ Output: {result['output'][:100]}...")
            
            time.sleep(1)  # Rate limiting
            
    finally:
        agent.shutdown()
    
    # Generate report
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['total_time_ms'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"\nSuccess Rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"Average Time: {avg_time:.0f}ms")
    print(f"Total Time: {total_time:.0f}ms")
    
    # Save detailed results
    output_dir = Path(__file__).parent / "results" / "proper_code_mode_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_queries": len(results),
            "successful": successful,
            "avg_time_ms": avg_time,
            "total_time_ms": total_time,
            "queries": results
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    # Summary report
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PROPER CODE MODE BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Architecture: Deno + TypeScript + HTTP Proxy\n")
        f.write(f"LLM: OpenAI GPT-4\n\n")
        f.write(f"Total Queries: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Success Rate: {100*successful/len(results):.1f}%\n")
        f.write(f"Average Time: {avg_time:.0f}ms\n")
        f.write(f"Total Time: {total_time:.0f}ms\n\n")
        
        f.write("Query Results:\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"\n{i}. {r['query']}\n")
            f.write(f"   Success: {r['success']}\n")
            f.write(f"   Time: {r['total_time_ms']:.0f}ms\n")
            if r['success']:
                try:
                    output_data = json.loads(r['output'])
                    num_results = len(output_data.get('results', []))
                    f.write(f"   Tools Found: {num_results}\n")
                except:
                    pass
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
