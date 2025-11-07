#!/usr/bin/env python3
"""
Comprehensive Benchmark: Traditional vs Code Mode on Full MetaTool Evaluation Set

Runs comparative benchmarks on all MetaTool evaluation datasets:
- Task1.json: Binary classification (tool needed: yes/no)
- Task2-Subtask1.json: Single tool selection
- Task2-Subtask2.json: Tool selection with categories
- Task2-Subtask3.json: Multi-step tool selection
- Task2-Subtask4.json: Complex tool reasoning

For each task, compares:
- Traditional tool calling (send ALL function schemas)
- Code Mode (send TypeScript API once)

Metrics tracked:
- Token usage (prompt, completion, total)
- Latency (per query, total)
- LLM call count
- Success rate
- Cost estimation
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmark_comparative import (
    BenchmarkResult,
    TraditionalToolCaller,
    CodeModeWrapper,
    load_metatool_database
)
from code_mode_proper import CodeModeAgent
from openai_integration import OpenAICodeGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskBenchmarkResult:
    """Results for a single task file"""
    task_name: str
    total_queries: int
    queries_processed: int
    
    # Traditional approach metrics
    traditional_total_tokens: int
    traditional_prompt_tokens: int
    traditional_completion_tokens: int
    traditional_total_time_ms: float
    traditional_avg_time_ms: float
    traditional_llm_calls: int
    traditional_success_count: int
    traditional_error_count: int
    
    # Code Mode metrics
    codemode_total_tokens: int
    codemode_prompt_tokens: int
    codemode_completion_tokens: int
    codemode_total_time_ms: float
    codemode_avg_time_ms: float
    codemode_llm_calls: int
    codemode_success_count: int
    codemode_error_count: int
    
    # Comparisons
    token_reduction_pct: float
    time_reduction_pct: float
    llm_call_reduction_pct: float
    
    # Cost estimates (GPT-4-turbo pricing: $10/1M prompt, $30/1M completion)
    traditional_cost_usd: float
    codemode_cost_usd: float
    cost_savings_pct: float


@dataclass
class FullBenchmarkSummary:
    """Overall summary across all tasks"""
    timestamp: str
    num_tools: int
    total_queries_all_tasks: int
    
    # Aggregate metrics
    traditional_total_tokens: int
    codemode_total_tokens: int
    token_reduction_pct: float
    
    traditional_total_time_ms: float
    codemode_total_time_ms: float
    time_reduction_pct: float
    
    traditional_total_llm_calls: int
    codemode_total_llm_calls: int
    llm_call_reduction_pct: float
    
    traditional_total_cost_usd: float
    codemode_total_cost_usd: float
    cost_savings_usd: float
    cost_savings_pct: float
    
    # Per-task results
    task_results: List[Dict[str, Any]]


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD using GPT-4-turbo pricing"""
    # GPT-4-turbo: $10/1M prompt tokens, $30/1M completion tokens
    prompt_cost = (prompt_tokens / 1_000_000) * 10
    completion_cost = (completion_tokens / 1_000_000) * 30
    return prompt_cost + completion_cost


def load_task_dataset(task_file: str) -> List[Dict[str, Any]]:
    """Load queries from a task file"""
    task_path = Path(__file__).parent.parent.parent / "dataset" / "tmp_dataset" / task_file
    
    if not task_path.exists():
        logger.error(f"Task file not found: {task_path}")
        raise FileNotFoundError(f"Task file not found: {task_path}")
    
    with open(task_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} queries from {task_file}")
    return data


def benchmark_task(
    task_name: str,
    task_file: str,
    tool_database,
    traditional_caller: TraditionalToolCaller,
    codemode_wrapper: CodeModeWrapper,
    sample_size: Optional[int] = None
) -> TaskBenchmarkResult:
    """
    Run benchmark on a single task file
    
    Args:
        task_name: Name of the task
        task_file: Filename of the task dataset
        tool_database: MetaTool database
        traditional_caller: Traditional approach caller
        codemode_wrapper: Code Mode wrapper
        sample_size: If set, only process this many queries (for testing)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARKING TASK: {task_name}")
    logger.info(f"{'='*80}\n")
    
    # Load task queries
    queries_data = load_task_dataset(task_file)
    if not queries_data:
        logger.error(f"No queries loaded for {task_name}")
        raise ValueError(f"No queries loaded for {task_name}")
    
    # Sample if requested
    if sample_size and sample_size < len(queries_data):
        logger.info(f"Sampling {sample_size} queries from {len(queries_data)} total")
        queries_data = queries_data[:sample_size]
    
    total_queries = len(queries_data)
    
    # Initialize metrics
    traditional_metrics = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_time_ms': 0.0,
        'llm_calls': 0,
        'success': 0,
        'errors': 0
    }
    
    codemode_metrics = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_time_ms': 0.0,
        'llm_calls': 0,
        'success': 0,
        'errors': 0
    }
    
    # Process each query
    for idx, query_data in enumerate(queries_data):
        query = query_data.get('query', '')
        if not query:
            logger.warning(f"Query {idx} has no query text, skipping")
            continue
        
        logger.info(f"\n[{idx+1}/{total_queries}] Query: {query[:100]}...")
        
        # Traditional approach
        logger.info("  Running TRADITIONAL approach...")
        try:
            trad_result = traditional_caller.process_query(query)
            traditional_metrics['total_tokens'] += trad_result.total_tokens
            traditional_metrics['prompt_tokens'] += trad_result.prompt_tokens
            traditional_metrics['completion_tokens'] += trad_result.completion_tokens
            traditional_metrics['total_time_ms'] += trad_result.latency_ms
            traditional_metrics['llm_calls'] += trad_result.num_llm_calls
            traditional_metrics['success'] += 1
            
            logger.info(f"  ✓ Traditional: {trad_result.total_tokens} tokens, "
                       f"{trad_result.latency_ms:.0f}ms, "
                       f"{trad_result.num_llm_calls} LLM calls")
        except Exception as e:
            logger.error(f"  ✗ Traditional failed: {e}")
            traditional_metrics['errors'] += 1
        
        # Code Mode approach
        logger.info("  Running CODE MODE approach...")
        try:
            cm_result = codemode_wrapper.process_query(query)
            codemode_metrics['total_tokens'] += cm_result.total_tokens
            codemode_metrics['prompt_tokens'] += cm_result.prompt_tokens
            codemode_metrics['completion_tokens'] += cm_result.completion_tokens
            codemode_metrics['total_time_ms'] += cm_result.latency_ms
            codemode_metrics['llm_calls'] += cm_result.num_llm_calls
            codemode_metrics['success'] += 1
            
            logger.info(f"  ✓ Code Mode: {cm_result.total_tokens} tokens, "
                       f"{cm_result.latency_ms:.0f}ms, "
                       f"{cm_result.num_llm_calls} LLM calls")
        except Exception as e:
            logger.error(f"  ✗ Code Mode failed: {e}")
            codemode_metrics['errors'] += 1
        
        # Show comparison
        if traditional_metrics['success'] > 0 and codemode_metrics['success'] > 0:
            token_savings = ((traditional_metrics['total_tokens'] - codemode_metrics['total_tokens']) / 
                           traditional_metrics['total_tokens'] * 100)
            logger.info(f"  📊 Token reduction so far: {token_savings:.1f}%")
    
    # Calculate averages and percentages
    queries_processed = traditional_metrics['success']  # Use traditional success count
    
    trad_avg_time = traditional_metrics['total_time_ms'] / max(queries_processed, 1)
    cm_avg_time = codemode_metrics['total_time_ms'] / max(queries_processed, 1)
    
    token_reduction = ((traditional_metrics['total_tokens'] - codemode_metrics['total_tokens']) / 
                      max(traditional_metrics['total_tokens'], 1) * 100)
    
    time_reduction = ((traditional_metrics['total_time_ms'] - codemode_metrics['total_time_ms']) / 
                     max(traditional_metrics['total_time_ms'], 1) * 100)
    
    llm_call_reduction = ((traditional_metrics['llm_calls'] - codemode_metrics['llm_calls']) / 
                         max(traditional_metrics['llm_calls'], 1) * 100)
    
    # Calculate costs
    trad_cost = calculate_cost(traditional_metrics['prompt_tokens'], 
                               traditional_metrics['completion_tokens'])
    cm_cost = calculate_cost(codemode_metrics['prompt_tokens'], 
                            codemode_metrics['completion_tokens'])
    cost_savings = ((trad_cost - cm_cost) / max(trad_cost, 0.001) * 100)
    
    # Create result
    result = TaskBenchmarkResult(
        task_name=task_name,
        total_queries=total_queries,
        queries_processed=queries_processed,
        
        traditional_total_tokens=traditional_metrics['total_tokens'],
        traditional_prompt_tokens=traditional_metrics['prompt_tokens'],
        traditional_completion_tokens=traditional_metrics['completion_tokens'],
        traditional_total_time_ms=traditional_metrics['total_time_ms'],
        traditional_avg_time_ms=trad_avg_time,
        traditional_llm_calls=traditional_metrics['llm_calls'],
        traditional_success_count=traditional_metrics['success'],
        traditional_error_count=traditional_metrics['errors'],
        
        codemode_total_tokens=codemode_metrics['total_tokens'],
        codemode_prompt_tokens=codemode_metrics['prompt_tokens'],
        codemode_completion_tokens=codemode_metrics['completion_tokens'],
        codemode_total_time_ms=codemode_metrics['total_time_ms'],
        codemode_avg_time_ms=cm_avg_time,
        codemode_llm_calls=codemode_metrics['llm_calls'],
        codemode_success_count=codemode_metrics['success'],
        codemode_error_count=codemode_metrics['errors'],
        
        token_reduction_pct=token_reduction,
        time_reduction_pct=time_reduction,
        llm_call_reduction_pct=llm_call_reduction,
        
        traditional_cost_usd=trad_cost,
        codemode_cost_usd=cm_cost,
        cost_savings_pct=cost_savings
    )
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"TASK SUMMARY: {task_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Queries processed: {queries_processed}/{total_queries}")
    logger.info(f"\nTraditional Approach:")
    logger.info(f"  Tokens: {traditional_metrics['total_tokens']:,} "
               f"(prompt: {traditional_metrics['prompt_tokens']:,}, "
               f"completion: {traditional_metrics['completion_tokens']:,})")
    logger.info(f"  Time: {traditional_metrics['total_time_ms']:.0f}ms "
               f"(avg: {trad_avg_time:.0f}ms/query)")
    logger.info(f"  LLM calls: {traditional_metrics['llm_calls']}")
    logger.info(f"  Cost: ${trad_cost:.4f}")
    logger.info(f"\nCode Mode Approach:")
    logger.info(f"  Tokens: {codemode_metrics['total_tokens']:,} "
               f"(prompt: {codemode_metrics['prompt_tokens']:,}, "
               f"completion: {codemode_metrics['completion_tokens']:,})")
    logger.info(f"  Time: {codemode_metrics['total_time_ms']:.0f}ms "
               f"(avg: {cm_avg_time:.0f}ms/query)")
    logger.info(f"  LLM calls: {codemode_metrics['llm_calls']}")
    logger.info(f"  Cost: ${cm_cost:.4f}")
    logger.info(f"\n🎯 IMPROVEMENTS:")
    logger.info(f"  Token reduction: {token_reduction:.1f}%")
    logger.info(f"  Time reduction: {time_reduction:.1f}%")
    logger.info(f"  LLM call reduction: {llm_call_reduction:.1f}%")
    logger.info(f"  Cost savings: {cost_savings:.1f}% (${trad_cost - cm_cost:.4f})")
    logger.info(f"{'='*80}\n")
    
    return result


def run_full_benchmark(
    num_tools: int = 100,
    sample_size: Optional[int] = None
) -> FullBenchmarkSummary:
    """
    Run comprehensive benchmark on all MetaTool evaluation tasks
    
    Args:
        num_tools: Number of tools to load from database
        sample_size: If set, only process this many queries per task (for testing)
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"FULL MetaTool EVALUATION BENCHMARK")
    logger.info(f"{'#'*80}")
    logger.info(f"Number of tools: {num_tools}")
    logger.info(f"Sample size per task: {sample_size if sample_size else 'ALL'}")
    logger.info(f"{'#'*80}\n")
    
    # Load tool database
    logger.info("Loading MetaTool database...")
    tool_database = load_metatool_database(num_tools=num_tools)
    logger.info(f"Loaded {len(tool_database.tools)} tools")
    
    # Initialize callers
    logger.info("Initializing Traditional and Code Mode callers...")
    traditional_caller = TraditionalToolCaller(tool_database)
    
    # Initialize Code Mode agent and wrapper
    llm_client = OpenAICodeGenerator(model="gpt-4")
    code_mode_agent = CodeModeAgent(
        tool_database=tool_database,
        llm_client=llm_client,
        proxy_port=3002  # Different port to avoid conflicts with manual testing
    )
    codemode_wrapper = CodeModeWrapper(code_mode_agent, llm_client)
    
    # Define tasks to benchmark
    tasks = [
        ("Task1 (Binary Classification)", "Task1.json"),
        ("Task2-Subtask1 (Single Tool Selection)", "Task2-Subtask1.json"),
        ("Task2-Subtask2 (Tool Selection with Categories)", "Task2-Subtask2.json"),
        ("Task2-Subtask3 (Multi-step Tool Selection)", "Task2-Subtask3.json"),
        ("Task2-Subtask4 (Complex Tool Reasoning)", "Task2-Subtask4.json"),
    ]
    
    # Run benchmarks
    task_results = []
    for task_name, task_file in tasks:
        result = benchmark_task(
            task_name=task_name,
            task_file=task_file,
            tool_database=tool_database,
            traditional_caller=traditional_caller,
            codemode_wrapper=codemode_wrapper,
            sample_size=sample_size
        )
        if result:
            task_results.append(result)
    
    # Aggregate results
    total_queries = sum(r.queries_processed for r in task_results)
    
    trad_total_tokens = sum(r.traditional_total_tokens for r in task_results)
    trad_prompt_tokens = sum(r.traditional_prompt_tokens for r in task_results)
    trad_completion_tokens = sum(r.traditional_completion_tokens for r in task_results)
    trad_total_time = sum(r.traditional_total_time_ms for r in task_results)
    trad_llm_calls = sum(r.traditional_llm_calls for r in task_results)
    trad_total_cost = sum(r.traditional_cost_usd for r in task_results)
    
    cm_total_tokens = sum(r.codemode_total_tokens for r in task_results)
    cm_prompt_tokens = sum(r.codemode_prompt_tokens for r in task_results)
    cm_completion_tokens = sum(r.codemode_completion_tokens for r in task_results)
    cm_total_time = sum(r.codemode_total_time_ms for r in task_results)
    cm_llm_calls = sum(r.codemode_llm_calls for r in task_results)
    cm_total_cost = sum(r.codemode_cost_usd for r in task_results)
    
    # Calculate overall reductions
    token_reduction = ((trad_total_tokens - cm_total_tokens) / max(trad_total_tokens, 1) * 100)
    time_reduction = ((trad_total_time - cm_total_time) / max(trad_total_time, 1) * 100)
    llm_call_reduction = ((trad_llm_calls - cm_llm_calls) / max(trad_llm_calls, 1) * 100)
    cost_savings_pct = ((trad_total_cost - cm_total_cost) / max(trad_total_cost, 0.001) * 100)
    
    # Create summary
    summary = FullBenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        num_tools=num_tools,
        total_queries_all_tasks=total_queries,
        
        traditional_total_tokens=trad_total_tokens,
        codemode_total_tokens=cm_total_tokens,
        token_reduction_pct=token_reduction,
        
        traditional_total_time_ms=trad_total_time,
        codemode_total_time_ms=cm_total_time,
        time_reduction_pct=time_reduction,
        
        traditional_total_llm_calls=trad_llm_calls,
        codemode_total_llm_calls=cm_llm_calls,
        llm_call_reduction_pct=llm_call_reduction,
        
        traditional_total_cost_usd=trad_total_cost,
        codemode_total_cost_usd=cm_total_cost,
        cost_savings_usd=trad_total_cost - cm_total_cost,
        cost_savings_pct=cost_savings_pct,
        
        task_results=[asdict(r) for r in task_results]
    )
    
    # Print overall summary
    logger.info(f"\n{'#'*80}")
    logger.info(f"OVERALL BENCHMARK SUMMARY")
    logger.info(f"{'#'*80}")
    logger.info(f"Total queries processed: {total_queries}")
    logger.info(f"Number of tools: {num_tools}")
    logger.info(f"\nTraditional Approach (TOTAL):")
    logger.info(f"  Tokens: {trad_total_tokens:,} (prompt: {trad_prompt_tokens:,}, completion: {trad_completion_tokens:,})")
    logger.info(f"  Time: {trad_total_time:.0f}ms ({trad_total_time/1000:.1f}s)")
    logger.info(f"  LLM calls: {trad_llm_calls}")
    logger.info(f"  Cost: ${trad_total_cost:.2f}")
    logger.info(f"\nCode Mode Approach (TOTAL):")
    logger.info(f"  Tokens: {cm_total_tokens:,} (prompt: {cm_prompt_tokens:,}, completion: {cm_completion_tokens:,})")
    logger.info(f"  Time: {cm_total_time:.0f}ms ({cm_total_time/1000:.1f}s)")
    logger.info(f"  LLM calls: {cm_llm_calls}")
    logger.info(f"  Cost: ${cm_total_cost:.2f}")
    logger.info(f"\n🎯 OVERALL IMPROVEMENTS:")
    logger.info(f"  Token reduction: {token_reduction:.1f}%")
    logger.info(f"  Time reduction: {time_reduction:.1f}%")
    logger.info(f"  LLM call reduction: {llm_call_reduction:.1f}%")
    logger.info(f"  Cost savings: {cost_savings_pct:.1f}% (${trad_total_cost - cm_total_cost:.2f})")
    logger.info(f"{'#'*80}\n")
    
    return summary


def save_results(summary: FullBenchmarkSummary, output_dir: str = "results/full_benchmark"):
    """Save benchmark results to JSON file"""
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"full_benchmark_{summary.num_tools}tools_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full MetaTool evaluation benchmark")
    parser.add_argument("--num-tools", type=int, default=100,
                       help="Number of tools to load (default: 100)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size per task for testing (default: all queries)")
    parser.add_argument("--output-dir", type=str, default="results/full_benchmark",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Run benchmark
        summary = run_full_benchmark(
            num_tools=args.num_tools,
            sample_size=args.sample_size
        )
        
        # Save results
        output_file = save_results(summary, args.output_dir)
        
        logger.info(f"\n✅ Benchmark completed successfully!")
        logger.info(f"📊 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Benchmark failed: {e}", exc_info=True)
        sys.exit(1)
