"""
Test proper Code Mode implementation with Deno
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from code_mode_proper import CodeModeAgent
from gateway_dispatcher import ToolDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class SimpleMockLLM:
    """Simple mock LLM that generates basic TypeScript code"""
    
    def __call__(self, prompt: str) -> str:
        """Generate TypeScript code to search for translation tools"""
        return """
const result = await searchTools({
  query: "Spanish to English translation",
  limit: 2
});

console.log(JSON.stringify(result, null, 2));
"""

def create_test_tools():
    """Create test tools"""
    tools = [
        {
            "name": "GoogleTranslate",
            "description": "Translate text from Spanish to English",
            "category": "translation",
            "api_endpoint": "https://translate.google.com/api"
        },
        {
            "name": "DeepL",
            "description": "Professional translation service for Spanish to English",
            "category": "translation", 
            "api_endpoint": "https://api.deepl.com"
        },
    ]
    return tools


def main():
    print("=" * 80)
    print("PROPER CODE MODE TEST (with Deno)")
    print("=" * 80)
    print()
    
    # Create tool database
    print("→ Creating tool database...")
    tools = create_test_tools()
    tool_db = ToolDatabase(tools)
    print(f"✓ Created {len(tools)} test tools\n")
    
    # Create Code Mode agent
    # Note: Using mock LLM for now - will replace with OpenAI once we verify Deno works
    print("→ Initializing Code Mode agent...")
    llm_client = SimpleMockLLM()
    
    agent = CodeModeAgent(
        tool_database=tool_db,
        llm_client=llm_client,
        proxy_port=3001
    )
    
    if not agent.deno_available:
        print("\n" + "!" * 80)
        print("ERROR: Deno is not installed!")
        print("!" * 80)
        print()
        print("Install Deno to use proper Code Mode:")
        print("  macOS/Linux: curl -fsSL https://deno.land/install.sh | sh")
        print("  Windows: irm https://deno.land/install.ps1 | iex")
        print()
        print("After installation, add to PATH and restart terminal")
        print("!" * 80)
        agent.shutdown()
        return
        
    print("✓ Deno runtime available\n")
    
    # Test query
    query = "Find a tool to translate Spanish to English"
    print(f"📝 Test query: {query}\n")
    
    print("=" * 80)
    print("EXECUTING CODE MODE")
    print("=" * 80)
    
    try:
        result = agent.process_query(query)
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Success: {result['success']}")
        print(f"Total time: {result['total_time_ms']:.0f}ms")
        print()
        
        if result['success']:
            print("Generated TypeScript code:")
            print("-" * 80)
            print(result['code'])
            print("-" * 80)
            print()
            print("Output:")
            print("-" * 80)
            print(result['output'])
            print("-" * 80)
        else:
            print(f"Error: {result['error']}")
            print()
            print("Generated code:")
            print("-" * 80)
            print(result['code'])
            print("-" * 80)
            
    finally:
        agent.shutdown()
        print("\n✓ Test complete")


if __name__ == "__main__":
    main()
