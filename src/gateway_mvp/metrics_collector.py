"""
Metrics Collection and Comparison Framework

Comprehensive framework for measuring and comparing:
- Token usage
- Latency
- Retrieval accuracy (Recall@K, Precision@1, nDCG)
- Cost
- Success rates
"""

import json
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import statistics


@dataclass
class Metrics:
    """Comprehensive metrics for tool retrieval evaluation."""
    # Token metrics
    avg_tokens_per_query: float
    total_tokens: int
    min_tokens: int
    max_tokens: int
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float
    total_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Accuracy metrics
    precision_at_1: float  # Did we get the right tool as top result?
    recall_at_5: float     # Is the right tool in top 5?
    ndcg_at_5: float       # Normalized Discounted Cumulative Gain
    success_rate: float    # % of queries that succeeded
    
    # Cost metrics (assuming GPT-4 pricing)
    cost_per_query_usd: float
    cost_per_1k_queries_usd: float
    
    # Additional metadata
    total_queries: int
    failed_queries: int
    approach: str  # "baseline" or "gateway"


@dataclass
class ComparisonResult:
    """Result of comparing two approaches."""
    baseline_metrics: Metrics
    gateway_metrics: Metrics
    improvements: Dict[str, float]  # % improvement for each metric
    summary: str


class MetricsCollector:
    """Collects and calculates metrics for tool retrieval experiments."""
    
    # Pricing (as of 2024, GPT-4 Turbo)
    COST_PER_1K_INPUT_TOKENS = 0.01  # $0.01 per 1K input tokens
    COST_PER_1K_OUTPUT_TOKENS = 0.03  # $0.03 per 1K output tokens
    
    def __init__(self):
        """Initialize metrics collector."""
        self.results = []
        self.ground_truth = {}  # Maps query to expected tool(s)
    
    def set_ground_truth(self, ground_truth: Dict[str, List[str]]):
        """
        Set ground truth for accuracy evaluation.
        
        Args:
            ground_truth: Dict mapping query to list of correct tool names
                         Example: {"translate query": ["translator_tool", "language_tool"]}
        """
        self.ground_truth = ground_truth
    
    def add_result(self, query: str, result: Any, approach: str):
        """
        Add a result for metric calculation.
        
        Args:
            query: User query
            result: Result object (BaselineResult or MetaToolResult)
            approach: "baseline" or "gateway"
        """
        self.results.append({
            "query": query,
            "result": result,
            "approach": approach
        })
    
    def calculate_precision_at_k(self, predicted: List[str], actual: List[str], k: int = 1) -> float:
        """
        Calculate Precision@K: What fraction of top-K predictions are correct?
        """
        if not predicted or not actual:
            return 0.0
        
        top_k = predicted[:k]
        correct = sum(1 for pred in top_k if pred in actual)
        return correct / k
    
    def calculate_recall_at_k(self, predicted: List[str], actual: List[str], k: int = 5) -> float:
        """
        Calculate Recall@K: What fraction of correct tools are in top-K?
        """
        if not actual:
            return 0.0
        
        top_k = predicted[:k]
        found = sum(1 for act in actual if act in top_k)
        return found / len(actual)
    
    def calculate_ndcg_at_k(self, predicted: List[str], actual: List[str], k: int = 5) -> float:
        """
        Calculate nDCG@K (Normalized Discounted Cumulative Gain).
        Measures ranking quality with position-based discounting.
        """
        if not predicted or not actual:
            return 0.0
        
        # DCG: Sum of (relevance / log2(position + 1))
        dcg = 0.0
        for i, pred in enumerate(predicted[:k]):
            if pred in actual:
                relevance = 1.0
                position = i + 1
                dcg += relevance / (2.0 ** (position - 1).bit_length())
        
        # IDCG: Best possible DCG
        idcg = sum(1.0 / (2.0 ** i.bit_length()) for i in range(min(len(actual), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_metrics(self, approach: str) -> Metrics:
        """
        Calculate comprehensive metrics for a specific approach.
        
        Args:
            approach: "baseline" or "gateway"
        
        Returns:
            Metrics object with all calculated metrics
        """
        # Filter results for this approach
        approach_results = [r for r in self.results if r["approach"] == approach]
        
        if not approach_results:
            raise ValueError(f"No results found for approach: {approach}")
        
        # Extract data
        tokens = []
        latencies = []
        successes = []
        precisions = []
        recalls = []
        ndcgs = []
        
        for item in approach_results:
            result = item["result"]
            query = item["query"]
            
            # Token metrics
            tokens_used = getattr(result, 'tokens_used', 0)
            tokens.append(tokens_used)
            
            # Latency metrics
            latency = getattr(result, 'execution_time_ms', 0.0)
            latencies.append(latency)
            
            # Success
            success = getattr(result, 'success', False)
            successes.append(1 if success else 0)
            
            # Accuracy metrics (if ground truth available)
            if query in self.ground_truth and success:
                actual_tools = self.ground_truth[query]
                
                # Get predicted tool(s)
                if approach == "baseline":
                    selected = getattr(result, 'selected_tool', None)
                    predicted = [selected['name']] if selected else []
                else:  # gateway
                    res = getattr(result, 'result', None)
                    if isinstance(res, dict):
                        predicted = [res.get('name', '')]
                    elif isinstance(res, list):
                        predicted = [t.get('name', '') for t in res if isinstance(t, dict)]
                    else:
                        predicted = []
                
                # Calculate metrics
                precisions.append(self.calculate_precision_at_k(predicted, actual_tools, k=1))
                recalls.append(self.calculate_recall_at_k(predicted, actual_tools, k=5))
                ndcgs.append(self.calculate_ndcg_at_k(predicted, actual_tools, k=5))
        
        # Calculate aggregate metrics
        total_queries = len(approach_results)
        failed_queries = total_queries - sum(successes)
        
        # Token stats
        avg_tokens = statistics.mean(tokens) if tokens else 0
        total_tokens_sum = sum(tokens)
        min_tokens = min(tokens) if tokens else 0
        max_tokens = max(tokens) if tokens else 0
        
        # Latency stats
        avg_latency = statistics.mean(latencies) if latencies else 0
        total_latency = sum(latencies)
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        sorted_latencies = sorted(latencies)
        p50_latency = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0
        
        # Accuracy stats
        precision_1 = statistics.mean(precisions) if precisions else 0.0
        recall_5 = statistics.mean(recalls) if recalls else 0.0
        ndcg_5 = statistics.mean(ndcgs) if ndcgs else 0.0
        success_rate = sum(successes) / total_queries if total_queries > 0 else 0.0
        
        # Cost calculation (simplified: assuming all tokens are input for now)
        cost_per_query = (avg_tokens / 1000) * self.COST_PER_1K_INPUT_TOKENS
        cost_per_1k = cost_per_query * 1000
        
        return Metrics(
            avg_tokens_per_query=avg_tokens,
            total_tokens=total_tokens_sum,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            avg_latency_ms=avg_latency,
            total_latency_ms=total_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            precision_at_1=precision_1,
            recall_at_5=recall_5,
            ndcg_at_5=ndcg_5,
            success_rate=success_rate,
            cost_per_query_usd=cost_per_query,
            cost_per_1k_queries_usd=cost_per_1k,
            total_queries=total_queries,
            failed_queries=failed_queries,
            approach=approach
        )
    
    def compare_approaches(self) -> ComparisonResult:
        """
        Compare baseline and gateway approaches.
        
        Returns:
            ComparisonResult with detailed comparison
        """
        baseline_metrics = self.calculate_metrics("baseline")
        gateway_metrics = self.calculate_metrics("gateway")
        
        # Calculate improvements (positive = gateway is better)
        improvements = {
            "token_reduction_pct": (
                (baseline_metrics.avg_tokens_per_query - gateway_metrics.avg_tokens_per_query) 
                / baseline_metrics.avg_tokens_per_query * 100
            ),
            "latency_reduction_pct": (
                (baseline_metrics.avg_latency_ms - gateway_metrics.avg_latency_ms) 
                / baseline_metrics.avg_latency_ms * 100
            ),
            "cost_reduction_pct": (
                (baseline_metrics.cost_per_query_usd - gateway_metrics.cost_per_query_usd) 
                / baseline_metrics.cost_per_query_usd * 100
            ),
            "precision_improvement_pct": (
                (gateway_metrics.precision_at_1 - baseline_metrics.precision_at_1) 
                / baseline_metrics.precision_at_1 * 100 if baseline_metrics.precision_at_1 > 0 else 0
            ),
            "recall_improvement_pct": (
                (gateway_metrics.recall_at_5 - baseline_metrics.recall_at_5) 
                / baseline_metrics.recall_at_5 * 100 if baseline_metrics.recall_at_5 > 0 else 0
            ),
        }
        
        # Generate summary
        summary = f"""
GATEWAY META-TOOL VS BASELINE COMPARISON
{'='*70}

TOKEN USAGE:
  Baseline: {baseline_metrics.avg_tokens_per_query:,.0f} tokens/query
  Gateway:  {gateway_metrics.avg_tokens_per_query:,.0f} tokens/query
  → REDUCTION: {improvements['token_reduction_pct']:.1f}%

LATENCY:
  Baseline: {baseline_metrics.avg_latency_ms:.2f}ms
  Gateway:  {gateway_metrics.avg_latency_ms:.2f}ms
  → REDUCTION: {improvements['latency_reduction_pct']:.1f}%

COST:
  Baseline: ${baseline_metrics.cost_per_query_usd:.6f}/query (${baseline_metrics.cost_per_1k_queries_usd:.2f}/1K queries)
  Gateway:  ${gateway_metrics.cost_per_query_usd:.6f}/query (${gateway_metrics.cost_per_1k_queries_usd:.2f}/1K queries)
  → REDUCTION: {improvements['cost_reduction_pct']:.1f}%

ACCURACY:
  Precision@1:
    Baseline: {baseline_metrics.precision_at_1:.3f}
    Gateway:  {gateway_metrics.precision_at_1:.3f}
    → CHANGE: {improvements['precision_improvement_pct']:+.1f}%
  
  Recall@5:
    Baseline: {baseline_metrics.recall_at_5:.3f}
    Gateway:  {gateway_metrics.recall_at_5:.3f}
    → CHANGE: {improvements['recall_improvement_pct']:+.1f}%
  
  nDCG@5:
    Baseline: {baseline_metrics.ndcg_at_5:.3f}
    Gateway:  {gateway_metrics.ndcg_at_5:.3f}

SUCCESS RATE:
  Baseline: {baseline_metrics.success_rate:.1%}
  Gateway:  {gateway_metrics.success_rate:.1%}

{'='*70}
"""
        
        return ComparisonResult(
            baseline_metrics=baseline_metrics,
            gateway_metrics=gateway_metrics,
            improvements=improvements,
            summary=summary
        )
    
    def export_results(self, filepath: str):
        """Export detailed results to JSON file."""
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }
        
        for item in self.results:
            result_dict = {
                "query": item["query"],
                "approach": item["approach"],
                "success": getattr(item["result"], "success", False),
                "tokens_used": getattr(item["result"], "tokens_used", 0),
                "execution_time_ms": getattr(item["result"], "execution_time_ms", 0.0),
            }
            
            # Add approach-specific details
            if item["approach"] == "baseline":
                selected = getattr(item["result"], "selected_tool", None)
                result_dict["selected_tool"] = selected["name"] if selected else None
            else:
                result_dict["code_generated"] = getattr(item["result"], "code_generated", "")
            
            export_data["results"].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Results exported to {filepath}")


def demo():
    """Demonstrate metrics collection."""
    print("="*70)
    print("METRICS COLLECTION DEMO")
    print("="*70)
    
    # This would normally be populated by actual experiments
    # For demo, we'll create mock data
    
    from baseline_evaluator import BaselineResult
    from gateway_dispatcher import MetaToolResult
    
    collector = MetricsCollector()
    
    # Set ground truth
    collector.set_ground_truth({
        "check air quality": ["airqualityforeast"],
        "translate text": ["MixerBox_Translate_AI_language_tutor", "translator"],
        "calculate math": ["calculator"]
    })
    
    # Mock baseline results (high token usage)
    baseline_results = [
        ("check air quality", BaselineResult(
            success=True,
            selected_tool={"name": "airqualityforeast"},
            execution_time_ms=8500.0,
            tokens_used=145000,
            prompt_sent="...",
            llm_response="airqualityforeast"
        )),
        ("translate text", BaselineResult(
            success=True,
            selected_tool={"name": "MixerBox_Translate_AI_language_tutor"},
            execution_time_ms=9200.0,
            tokens_used=148000,
            prompt_sent="...",
            llm_response="MixerBox_Translate_AI_language_tutor"
        )),
    ]
    
    # Mock gateway results (low token usage)
    gateway_results = [
        ("check air quality", MetaToolResult(
            success=True,
            result={"name": "airqualityforeast", "score": 0.95},
            execution_time_ms=1800.0,
            tokens_used=1950,
            code_generated="results = meta_tool_search(...)"
        )),
        ("translate text", MetaToolResult(
            success=True,
            result={"name": "MixerBox_Translate_AI_language_tutor", "score": 0.92},
            execution_time_ms=2100.0,
            tokens_used=2150,
            code_generated="results = meta_tool_search(...)"
        )),
    ]
    
    # Add results
    for query, result in baseline_results:
        collector.add_result(query, result, "baseline")
    
    for query, result in gateway_results:
        collector.add_result(query, result, "gateway")
    
    # Compare
    comparison = collector.compare_approaches()
    print(comparison.summary)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo()
