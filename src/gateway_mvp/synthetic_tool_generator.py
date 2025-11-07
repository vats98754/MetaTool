"""
Synthetic Tool Library Generator

Generates 10,000+ diverse tool schemas for testing gateway meta-tool compression.
Creates realistic tool definitions with varying complexity, categories, and parameters.
"""

import json
import random
import os
from typing import List, Dict, Any
from datetime import datetime


class SyntheticToolGenerator:
    """Generates synthetic tools for testing tool retrieval systems."""
    
    # Tool categories for realistic distribution
    CATEGORIES = [
        "translation", "weather", "finance", "travel", "education",
        "entertainment", "productivity", "health", "sports", "news",
        "social_media", "e-commerce", "data_analysis", "ai_ml", "iot",
        "security", "communication", "gaming", "music", "video",
        "food", "real_estate", "automotive", "legal", "marketing"
    ]
    
    # Common parameter types
    PARAM_TYPES = ["string", "integer", "boolean", "number", "array", "object"]
    
    # Action verbs for tool names
    ACTION_VERBS = [
        "search", "find", "get", "fetch", "retrieve", "analyze", "calculate",
        "convert", "translate", "check", "verify", "validate", "compare",
        "list", "show", "display", "monitor", "track", "predict", "estimate",
        "generate", "create", "build", "parse", "extract", "filter", "sort"
    ]
    
    # Domain-specific nouns
    DOMAIN_NOUNS = [
        "data", "information", "results", "statistics", "metrics", "reports",
        "forecasts", "predictions", "analysis", "trends", "patterns", "insights",
        "recommendations", "suggestions", "alerts", "notifications", "updates",
        "summaries", "details", "profiles", "records", "documents", "files"
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.generated_tools = []
    
    def generate_tool_name(self, category: str) -> str:
        """Generate a realistic tool name."""
        verb = random.choice(self.ACTION_VERBS)
        noun = random.choice(self.DOMAIN_NOUNS)
        category_prefix = category.replace("_", "")
        
        # Various naming patterns
        patterns = [
            f"{verb}_{category_prefix}_{noun}",
            f"{category_prefix}_{verb}_{noun}",
            f"{verb}_{noun}_{category_prefix}",
            f"{category_prefix}{verb.capitalize()}{noun.capitalize()}",
            f"{verb.capitalize()}{category_prefix.capitalize()}",
        ]
        
        return random.choice(patterns)
    
    def generate_description(self, category: str, tool_name: str) -> str:
        """Generate a realistic tool description."""
        templates = [
            f"A powerful tool to {random.choice(self.ACTION_VERBS)} {category}-related {random.choice(self.DOMAIN_NOUNS)}. "
            f"Perfect for users who need {category} capabilities.",
            
            f"Get instant access to {category} {random.choice(self.DOMAIN_NOUNS)}. "
            f"This tool helps you {random.choice(self.ACTION_VERBS)} and analyze {category} data efficiently.",
            
            f"{tool_name} provides comprehensive {category} services including "
            f"{random.choice(self.ACTION_VERBS)}ing and {random.choice(self.ACTION_VERBS)}ing capabilities.",
            
            f"Professional-grade {category} tool for {random.choice(self.ACTION_VERBS)}ing "
            f"{random.choice(self.DOMAIN_NOUNS)}. Trusted by thousands of users worldwide.",
            
            f"Streamline your {category} workflow with this all-in-one tool. "
            f"Features include {random.choice(self.ACTION_VERBS)}ing, {random.choice(self.ACTION_VERBS)}ing, "
            f"and advanced {random.choice(self.DOMAIN_NOUNS)} processing.",
        ]
        
        return random.choice(templates)
    
    def generate_parameters(self, complexity: str = "medium") -> Dict[str, Any]:
        """Generate tool parameters based on complexity level."""
        param_count = {
            "simple": random.randint(1, 3),
            "medium": random.randint(3, 6),
            "complex": random.randint(6, 12)
        }
        
        num_params = param_count.get(complexity, 4)
        parameters = {}
        
        common_param_names = [
            "query", "location", "date", "limit", "offset", "format",
            "language", "category", "filter", "sort_by", "api_key",
            "user_id", "start_date", "end_date", "country", "city",
            "zip_code", "radius", "min_value", "max_value", "threshold"
        ]
        
        for _ in range(num_params):
            param_name = random.choice(common_param_names)
            param_type = random.choice(self.PARAM_TYPES)
            
            param_def = {
                "type": param_type,
                "description": f"The {param_name} parameter for the operation",
                "required": random.choice([True, False])
            }
            
            # Add constraints based on type
            if param_type == "integer":
                param_def["minimum"] = random.choice([0, 1, 10])
                param_def["maximum"] = random.choice([100, 1000, 10000])
            elif param_type == "string":
                if random.random() > 0.7:
                    param_def["enum"] = [f"option_{i}" for i in range(random.randint(2, 5))]
            elif param_type == "array":
                param_def["items"] = {"type": random.choice(["string", "integer"])}
            
            parameters[param_name] = param_def
        
        return parameters
    
    def generate_tool_schema(self, tool_id: int) -> Dict[str, Any]:
        """Generate a complete tool schema."""
        category = random.choice(self.CATEGORIES)
        tool_name = self.generate_tool_name(category)
        description = self.generate_description(category, tool_name)
        complexity = random.choice(["simple", "medium", "complex"])
        parameters = self.generate_parameters(complexity)
        
        schema = {
            "id": tool_id,
            "name": tool_name,
            "category": category,
            "description": description,
            "complexity": complexity,
            "version": f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}",
            "parameters": parameters,
            "metadata": {
                "author": f"developer_{random.randint(1, 100)}",
                "created_at": datetime.now().isoformat(),
                "popularity_score": round(random.uniform(0.1, 1.0), 2),
                "usage_count": random.randint(10, 100000),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "supports_batch": random.choice([True, False]),
                "supports_async": random.choice([True, False]),
                "rate_limit": random.choice([100, 1000, 10000, -1]),
            }
        }
        
        return schema
    
    def generate_tool_library(self, num_tools: int = 10000, output_path: str = None) -> List[Dict[str, Any]]:
        """
        Generate a complete library of synthetic tools.
        
        Args:
            num_tools: Number of tools to generate
            output_path: Path to save the generated tools (optional)
        
        Returns:
            List of tool schemas
        """
        print(f"Generating {num_tools} synthetic tools...")
        
        tools = []
        for i in range(num_tools):
            tool = self.generate_tool_schema(i + 1)
            tools.append(tool)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_tools} tools...")
        
        self.generated_tools = tools
        
        if output_path:
            self._save_tools(tools, output_path)
        
        print(f"✓ Successfully generated {num_tools} tools")
        return tools
    
    def _save_tools(self, tools: List[Dict[str, Any]], output_path: str):
        """Save generated tools to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(tools, f, indent=2)
        
        print(f"✓ Saved tools to {output_path}")
    
    def generate_tool_embeddings_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the tool library for embedding systems."""
        if not self.generated_tools:
            return {}
        
        category_dist = {}
        complexity_dist = {"simple": 0, "medium": 0, "complex": 0}
        
        for tool in self.generated_tools:
            cat = tool["category"]
            category_dist[cat] = category_dist.get(cat, 0) + 1
            complexity_dist[tool["complexity"]] += 1
        
        return {
            "total_tools": len(self.generated_tools),
            "categories": list(category_dist.keys()),
            "category_distribution": category_dist,
            "complexity_distribution": complexity_dist,
            "avg_parameters": sum(len(t["parameters"]) for t in self.generated_tools) / len(self.generated_tools),
            "has_embeddings": True,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "supports_semantic_search": True
        }
    
    def generate_compact_tool_list(self) -> List[Dict[str, str]]:
        """Generate a compact version (name + description only) for baseline comparison."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "category": tool["category"]
            }
            for tool in self.generated_tools
        ]


def main():
    """Main execution function."""
    # Create generator
    generator = SyntheticToolGenerator(seed=42)
    
    # Output paths
    base_path = "dataset/synthetic_tools"
    full_tools_path = f"{base_path}/synthetic_tools_10k.json"
    compact_tools_path = f"{base_path}/synthetic_tools_compact.json"
    metadata_path = f"{base_path}/tool_library_metadata.json"
    
    # Generate 10,000 tools
    tools = generator.generate_tool_library(
        num_tools=10000,
        output_path=full_tools_path
    )
    
    # Generate compact version for baseline
    compact_tools = generator.generate_compact_tool_list()
    generator._save_tools(compact_tools, compact_tools_path)
    
    # Generate metadata
    metadata = generator.generate_tool_embeddings_metadata()
    generator._save_tools(metadata, metadata_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SYNTHETIC TOOL LIBRARY GENERATION COMPLETE")
    print("="*60)
    print(f"Total tools generated: {len(tools)}")
    print(f"Categories: {len(metadata['categories'])}")
    print(f"Average parameters per tool: {metadata['avg_parameters']:.2f}")
    print(f"\nComplexity distribution:")
    for complexity, count in metadata['complexity_distribution'].items():
        print(f"  {complexity}: {count} ({count/len(tools)*100:.1f}%)")
    
    print(f"\nTop 5 categories by count:")
    sorted_cats = sorted(metadata['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
    for cat, count in sorted_cats:
        print(f"  {cat}: {count}")
    
    print(f"\nFiles saved:")
    print(f"  - Full schemas: {full_tools_path}")
    print(f"  - Compact list: {compact_tools_path}")
    print(f"  - Metadata: {metadata_path}")
    print("="*60)


if __name__ == "__main__":
    main()
