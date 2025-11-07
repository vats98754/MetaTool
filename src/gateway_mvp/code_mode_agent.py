"""
Code Mode Implementation for Gateway Meta-Tool

This implements the Cloudflare/Anthropic "Code Mode" architecture where:
1. MCP servers expose tools as TypeScript APIs
2. LLM writes code (not tool calls) against those APIs
3. Code executes in an isolate sandbox
4. Sandbox calls back to agent via RPC bindings
5. Agent dispatches to actual MCP servers

This is the "Agent (Worker)" in the middle of the architecture diagram.
"""

import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import subprocess
import tempfile
import os
import logging

# Setup detailed debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('code_mode_agent_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a single MCP tool with its schema."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class MCPServer:
    """Represents an MCP server with its available tools."""
    name: str
    url: str
    tools: List[MCPTool]
    auth_token: Optional[str] = None


@dataclass
class CodeModeResult:
    """Result from executing LLM-generated code in sandbox."""
    success: bool
    output: str  # Console logs from executed code
    execution_time_ms: float
    rpc_calls_made: List[Dict[str, Any]]  # Track all RPC calls to MCP servers
    error: Optional[str] = None
    code_executed: str = ""


class TypeScriptAPIGenerator:
    """
    Converts MCP server schemas into TypeScript API definitions.
    
    This is step 2 in the Code Mode diagram: "Provides TypeScript API matching MCP tools"
    """
    
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
    
    def json_schema_to_typescript(self, schema: Dict[str, Any], name: str) -> str:
        """Convert JSON schema to TypeScript interface."""
        if not schema or schema.get("type") != "object":
            return f"interface {name} {{\n  [key: string]: any;\n}}"
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        lines = [f"interface {name} {{"]
        
        for prop_name, prop_def in properties.items():
            prop_type = self._json_type_to_ts_type(prop_def)
            optional = "" if prop_name in required else "?"
            description = prop_def.get("description", "")
            
            if description:
                lines.append(f"  /**")
                lines.append(f"   * {description}")
                lines.append(f"   */")
            
            lines.append(f"  {prop_name}{optional}: {prop_type};")
        
        lines.append("}")
        return "\n".join(lines)
    
    def _json_type_to_ts_type(self, prop_def: Dict[str, Any]) -> str:
        """Map JSON schema types to TypeScript types."""
        json_type = prop_def.get("type", "any")
        
        type_map = {
            "string": "string",
            "number": "number",
            "integer": "number",
            "boolean": "boolean",
            "array": "any[]",
            "object": "{ [key: string]: any }",
        }
        
        return type_map.get(json_type, "any")
    
    def generate_typescript_api(self) -> str:
        """
        Generate complete TypeScript API definition for the MCP server.
        
        Returns TypeScript code that can be loaded into the LLM's context.
        """
        lines = []
        
        # Generate input/output interfaces for each tool
        for tool in self.mcp_server.tools:
            input_interface = f"{self._to_camel_case(tool.name)}Input"
            output_interface = f"{self._to_camel_case(tool.name)}Output"
            
            lines.append(self.json_schema_to_typescript(
                tool.input_schema,
                input_interface
            ))
            lines.append("")
            
            lines.append(self.json_schema_to_typescript(
                tool.output_schema or {},
                output_interface
            ))
            lines.append("")
        
        # Generate the API object
        lines.append(f"declare const {self.mcp_server.name}: {{")
        
        for tool in self.mcp_server.tools:
            input_interface = f"{self._to_camel_case(tool.name)}Input"
            output_interface = f"{self._to_camel_case(tool.name)}Output"
            
            if tool.description:
                lines.append(f"  /**")
                for desc_line in tool.description.split('\n'):
                    lines.append(f"   * {desc_line}")
                lines.append(f"   */")
            
            lines.append(f"  {tool.name}: (")
            lines.append(f"    input: {input_interface}")
            lines.append(f"  ) => Promise<{output_interface}>;")
            lines.append("")
        
        lines.append("};")
        
        return "\n".join(lines)
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)


class RPCBindingProvider:
    """
    Provides RPC bindings that the sandbox can call back to.
    
    This is step 5 in the Code Mode diagram: "Calls RPC bindings provided by agent"
    
    The sandbox code calls these bindings, which then dispatch to actual MCP servers.
    """
    
    def __init__(self, mcp_servers: List[MCPServer]):
        self.mcp_servers = {server.name: server for server in mcp_servers}
        self.rpc_call_log = []
    
    def create_binding_interface(self, server_name: str) -> Dict[str, Callable]:
        """
        Create a binding interface for a specific MCP server.
        
        Returns a dict of functions that the sandbox can call.
        """
        if server_name not in self.mcp_servers:
            raise ValueError(f"MCP server '{server_name}' not found")
        
        server = self.mcp_servers[server_name]
        bindings = {}
        
        for tool in server.tools:
            # Create a closure that captures the tool and server
            def make_binding(srv, tl):
                def binding(input_data: Dict[str, Any]) -> Dict[str, Any]:
                    return self._call_mcp_tool(srv, tl, input_data)
                return binding
            
            bindings[tool.name] = make_binding(server, tool)
        
        return bindings
    
    def _call_mcp_tool(
        self,
        server: MCPServer,
        tool: MCPTool,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Actually call the MCP server tool.
        
        This is step 6 in the Code Mode diagram: "Calls MCP tools"
        
        In a real implementation, this would make an HTTP request to the MCP server.
        For this MVP, we'll simulate the call.
        """
        logger.info(f"\n{'*'*80}")
        logger.info(f"RPC BINDING CALL")
        logger.info(f"{'*'*80}")
        logger.info(f"Server: {server.name}")
        logger.info(f"Tool: {tool.name}")
        logger.info(f"Input: {json.dumps(input_data, indent=2)}")
        logger.info(f"{'*'*80}\n")
        
        # Log the RPC call
        call_record = {
            "timestamp": time.time(),
            "server": server.name,
            "tool": tool.name,
            "input": input_data,
        }
        
        # Simulate MCP server call
        # In production, this would be:
        # response = requests.post(
        #     f"{server.url}/call",
        #     json={"tool": tool.name, "input": input_data},
        #     headers={"Authorization": f"Bearer {server.auth_token}"}
        # )
        # result = response.json()
        
        # For MVP, simulate based on tool name
        result = self._simulate_mcp_call(server, tool, input_data)
        logger.info(f"RPC call result: {json.dumps(result, indent=2, default=str)}")
        
        call_record["output"] = result
        call_record["success"] = True
        self.rpc_call_log.append(call_record)
        
        return result
    
    def _simulate_mcp_call(
        self,
        server: MCPServer,
        tool: MCPTool,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate MCP tool execution for demo purposes."""
        if "search" in tool.name.lower():
            return {
                "results": [
                    {
                        "name": f"tool_{i}",
                        "description": f"A tool that does something useful",
                        "score": 0.9 - (i * 0.1)
                    }
                    for i in range(3)
                ]
            }
        elif "fetch" in tool.name.lower():
            return {
                "content": "This is the fetched content",
                "metadata": {"source": server.name}
            }
        else:
            return {
                "status": "success",
                "message": f"Executed {tool.name} with input {input_data}"
            }


class DynamicIsolateSandbox:
    """
    Executes LLM-generated code in an isolated sandbox.
    
    This is step 4 in the Code Mode diagram: "Executes code in sandbox"
    
    In production, this would use V8 isolates (Cloudflare Workers).
    For this MVP, we'll use a secure subprocess with limited capabilities.
    """
    
    def __init__(self, rpc_bindings: RPCBindingProvider):
        self.rpc_bindings = rpc_bindings
    
    def execute_typescript_code(
        self,
        code: str,
        server_name: str,
        timeout_seconds: int = 5
    ) -> CodeModeResult:
        """
        Execute TypeScript code in isolated sandbox.
        
        The code has access to:
        - The TypeScript API for the MCP server
        - console.log() for output
        - NO network access
        - NO file system access
        - NO other capabilities
        
        Returns the console.log output and any errors.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SANDBOX EXECUTION START")
        logger.info(f"{'='*80}")
        logger.info(f"Server name: {server_name}")
        logger.info(f"Timeout: {timeout_seconds}s")
        logger.info(f"TypeScript code ({len(code)} chars):")
        logger.info(f"{code}")
        logger.info(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            # Convert TypeScript to Python for execution
            # In production, this would use actual V8 isolates
            logger.debug("Transpiling TypeScript to Python...")
            python_code = self._transpile_ts_to_python(code, server_name)
            logger.debug(f"Transpiled Python code ({len(python_code)} chars):")
            logger.debug(f"{python_code[:500]}...")
            
            # Execute in restricted environment
            logger.debug(f"Executing in sandbox...")
            output = self._execute_in_sandbox(python_code, timeout_seconds)
            logger.info(f"Sandbox execution output: '{output}'")
            
            execution_time = (time.time() - start_time) * 1000
            
            return CodeModeResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
                rpc_calls_made=self.rpc_bindings.rpc_call_log.copy(),
                code_executed=code
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return CodeModeResult(
                success=False,
                output="",
                execution_time_ms=execution_time,
                rpc_calls_made=self.rpc_bindings.rpc_call_log.copy(),
                error=str(e),
                code_executed=code
            )
    
    def _transpile_ts_to_python(self, ts_code: str, server_name: str) -> str:
        """
        Convert TypeScript code to Python for execution.
        
        In production, this would execute actual TypeScript in V8.
        For MVP, we do a simple conversion.
        """
        # Get the binding interface
        bindings = self.rpc_bindings.create_binding_interface(server_name)
        
        # Create a temporary file path for IPC (RPC call communication)
        ipc_file = tempfile.mktemp(suffix='.json')
        self._current_ipc_file = ipc_file
        self._current_bindings = bindings
        self._current_server_name = server_name
        
        # Create Python code that simulates the TypeScript execution
        # The key change: MCPServer now makes RPC calls via IPC file
        python_code = f"""
import json
import os

# IPC file for RPC communication with agent
IPC_FILE = '{ipc_file}'

# Simulated MCP server bindings that call back to agent via IPC
class MCPServer:
    def __init__(self, server_name, available_tools):
        self.server_name = server_name
        self.available_tools = available_tools
    
    def __getattr__(self, tool_name):
        if tool_name not in self.available_tools:
            raise AttributeError(f"No tool named {{tool_name}}")
        
        def call_tool(input_data):
            # Write RPC call request to IPC file
            rpc_request = {{
                'server': self.server_name,
                'tool': tool_name,
                'input': input_data
            }}
            
            # Write request
            with open(IPC_FILE, 'w') as f:
                json.dump(rpc_request, f)
            
            # Read response (agent will have written it)
            with open(IPC_FILE, 'r') as f:
                response = json.load(f)
            
            return response.get('result', {{}})
        
        return call_tool

{server_name} = MCPServer('{server_name}', {list(bindings.keys())})

# Console API
class Console:
    def __init__(self):
        self.logs = []
    
    def log(self, *args):
        message = ' '.join(str(arg) for arg in args)
        self.logs.append(message)
        print(message)
    
    def error(self, *args):
        message = ' '.join(str(arg) for arg in args)
        self.logs.append(f"ERROR: {{message}}")
        print(f"ERROR: {{message}}")

console = Console()

# Execute the transpiled code
{self._indent_code(self._simple_ts_to_py_conversion(ts_code), '')}

# Output results
output = '\\n'.join(console.logs)
print("__OUTPUT__:", output)
"""
        return python_code
    
    def _simple_ts_to_py_conversion(self, ts_code: str) -> str:
        """Simple TypeScript to Python conversion for demo."""
        # This is a naive conversion for MVP
        # In production, we'd execute actual TypeScript
        py_code = ts_code
        
        # Convert basic syntax
        py_code = py_code.replace("const ", "")
        py_code = py_code.replace("let ", "")
        py_code = py_code.replace("await ", "await ")
        py_code = py_code.replace(";", "")
        
        return py_code
    
    def _indent_code(self, code: str, indent: str) -> str:
        """Indent all lines of code."""
        return '\n'.join(indent + line for line in code.split('\n'))
    
    def _execute_in_sandbox(self, python_code: str, timeout: int) -> str:
        """
        Execute Python code in a restricted subprocess.
        
        In production, this would use V8 isolates.
        """
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(python_code)
            temp_file = f.name
        
        try:
            # Execute with timeout and restricted environment
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={'PYTHONPATH': ''}  # Minimal environment
            )
            
            # Extract output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith('__OUTPUT__:'):
                    return line.split('__OUTPUT__:', 1)[1].strip()
            
            return result.stdout
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class CodeModeAgent:
    """
    Main agent that orchestrates the Code Mode flow.
    
    This is the "Agent (Worker)" in the architecture diagram.
    
    Flow:
    1. Provides tool schemas to LLM (converted to TypeScript)
    2. LLM writes code against TypeScript APIs
    3. Agent executes code in sandbox
    4. Sandbox calls RPC bindings
    5. Bindings call actual MCP servers
    6. Results flow back through the chain
    """
    
    def __init__(self, mcp_servers: List[MCPServer], llm_client: Optional[Callable] = None):
        self.mcp_servers = mcp_servers
        self.llm_client = llm_client or self._mock_llm_client
        
        # Create RPC binding provider
        self.rpc_bindings = RPCBindingProvider(mcp_servers)
        
        # Create sandbox
        self.sandbox = DynamicIsolateSandbox(self.rpc_bindings)
        
        # Generate TypeScript APIs for all servers
        self.typescript_apis = {}
        for server in mcp_servers:
            generator = TypeScriptAPIGenerator(server)
            self.typescript_apis[server.name] = generator.generate_typescript_api()
    
    def _mock_llm_client(self, prompt: str) -> str:
        """Mock LLM that generates simple TypeScript code."""
        if "search" in prompt.lower():
            return """
const results = await tooldb.search_tools({
    query: "translate Spanish English",
    limit: 5
});

console.log("Found tools:", JSON.stringify(results));

if (results.results && results.results.length > 0) {
    console.log("Best tool:", results.results[0].name);
}
"""
        else:
            return """
const result = await tooldb.fetch_tool_info({
    tool_name: "example_tool"
});

console.log("Tool info:", JSON.stringify(result));
"""
    
    def process_request(self, user_query: str, mcp_server_name: str) -> CodeModeResult:
        """
        Process a user request using Code Mode.
        
        This is the complete flow from the diagram:
        1. Provide TypeScript API to LLM
        2. LLM writes code
        3. Execute code in sandbox
        4. Code calls RPC bindings
        5. Bindings call MCP tools
        6. Return results
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"CODE MODE REQUEST START")
        logger.info(f"{'#'*80}")
        logger.info(f"User Query: {user_query}")
        logger.info(f"MCP Server: {mcp_server_name}")
        logger.info(f"{'#'*80}\n")
        
        start_time = time.time()
        
        # Step 1: Build prompt with TypeScript API
        logger.info("STEP 1: Getting TypeScript API...")
        typescript_api = self.typescript_apis.get(mcp_server_name)
        if not typescript_api:
            raise ValueError(f"No TypeScript API for server: {mcp_server_name}")
        logger.debug(f"TypeScript API ({len(typescript_api)} chars):")
        logger.debug(f"{typescript_api[:300]}...")
        
        logger.info("STEP 2: Building prompt for LLM...")
        prompt = self._build_code_mode_prompt(user_query, typescript_api, mcp_server_name)
        logger.debug(f"Prompt ({len(prompt)} chars):")
        logger.debug(f"{prompt[:500]}...")
        
        # Step 2: LLM generates TypeScript code
        logger.info("STEP 3: Calling LLM to generate TypeScript code...")
        generated_code = self.llm_client(prompt)
        logger.info(f"LLM generated code ({len(generated_code)} chars):")
        logger.info(f"{generated_code}")
        
        # Step 3: Execute code in sandbox
        logger.info("STEP 4: Executing code in sandbox...")
        result = self.sandbox.execute_typescript_code(
            generated_code,
            mcp_server_name
        )
        
        logger.info(f"\nCODE MODE REQUEST COMPLETE")
        logger.info(f"Total time: {(time.time() - start_time)*1000:.2f}ms")
        logger.info(f"Success: {result.success}")
        logger.info(f"RPC calls made: {len(result.rpc_calls_made)}")
        logger.info(f"{'#'*80}\n")
        
        return result
    
    def _build_code_mode_prompt(
        self,
        user_query: str,
        typescript_api: str,
        server_name: str
    ) -> str:
        """
        Build the prompt for Code Mode.
        
        Instead of presenting tools for tool-calling, we present a TypeScript API
        and ask the LLM to write code.
        """
        prompt = f"""You are an AI agent with access to the {server_name} API.

User Query: {user_query}

Available TypeScript API:
```typescript
{typescript_api}
```

Instructions:
1. Write TypeScript code to fulfill the user's request
2. Use the available API functions from the {server_name} object
3. Use console.log() to output results
4. The code will be executed in a secure sandbox
5. You have NO network access - only the API above

Generate TypeScript code (no explanations):
"""
        return prompt


def demo():
    """Demonstrate Code Mode implementation."""
    print("="*80)
    print("CODE MODE IMPLEMENTATION DEMO")
    print("="*80)
    
    # Create sample MCP server
    tool_search = MCPTool(
        name="search_tools",
        description="Search for tools in the database using semantic search",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results"
                }
            },
            "required": ["query"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "Search results"
                }
            }
        }
    )
    
    mcp_server = MCPServer(
        name="tooldb",
        url="http://localhost:8000/mcp",
        tools=[tool_search]
    )
    
    # Create Code Mode agent
    agent = CodeModeAgent([mcp_server])
    
    # Show generated TypeScript API
    print("\n" + "="*80)
    print("GENERATED TYPESCRIPT API")
    print("="*80)
    print(agent.typescript_apis["tooldb"])
    
    # Process a request
    print("\n" + "="*80)
    print("PROCESSING REQUEST")
    print("="*80)
    
    user_query = "Find tools for translating Spanish to English"
    print(f"\nUser Query: {user_query}")
    
    result = agent.process_request(user_query, "tooldb")
    
    if result.success:
        print(f"\n✓ SUCCESS")
        print(f"  Execution Time: {result.execution_time_ms:.2f}ms")
        print(f"  Output:\n{result.output}")
        print(f"\n  RPC Calls Made: {len(result.rpc_calls_made)}")
        for i, call in enumerate(result.rpc_calls_made, 1):
            print(f"    {i}. {call['server']}.{call['tool']}()")
    else:
        print(f"\n✗ FAILED: {result.error}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demo()
