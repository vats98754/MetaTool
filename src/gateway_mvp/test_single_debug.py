"""
Detailed Debug Test - Single Query
"""

import sys
import logging

# Setup logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('single_query_debug.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

from unified_gateway_codemode import UnifiedGatewayCodeMode
from gateway_dispatcher import ToolDatabase
from openai_integration import OpenAICodeGenerator

def main():
    print("="*80)
    print("SINGLE QUERY DEBUG TEST")
    print("="*80)
    
    # Create simple test tools
    test_tools = [
        {
            "id": 1,
            "name": "spanish_translator",
            "category": "translation",
            "description": "Translate text from Spanish to English",
            "parameters": {
                "text": {"type": "string", "required": True}
            }
        }
    ]
    
    print(f"\n✓ Created {len(test_tools)} test tools")
    
    # Create tool database
    tool_db = ToolDatabase(test_tools)
    print(f"✓ Tool database initialized")
    
    # Create OpenAI client
    print(f"\n🤖 Initializing OpenAI client...")
    llm_client = OpenAICodeGenerator(model="gpt-4")
    print(f"✓ OpenAI client ready")
    
    # Create unified agent
    print(f"\n🔧 Creating unified agent...")
    agent = UnifiedGatewayCodeMode(tool_db, llm_client=llm_client)
    print(f"✓ Agent initialized")
    
    # Run test query
    test_query = "Find a tool to translate Spanish to English"
    print(f"\n📝 Test query: {test_query}")
    print(f"\n{'='*80}")
    print(f"STARTING EXECUTION WITH FULL DEBUG LOGGING")
    print(f"{'='*80}\n")
    
    result = agent.process_query(test_query)
    
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {result.success}")
    print(f"Tool found: {result.selected_tool}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"RPC calls made: {len(result.rpc_calls)}")
    print(f"\nCode generated:")
    print(f"{'-'*80}")
    print(result.code_generated)
    print(f"{'-'*80}")
    print(f"\nConsole output:")
    print(f"{'-'*80}")
    print(result.console_output)
    print(f"{'-'*80}")
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
    
    print(f"\n✓ Detailed logs saved to: single_query_debug.log")
    print(f"✓ Agent logs saved to: code_mode_agent_debug.log")


if __name__ == "__main__":
    main()
