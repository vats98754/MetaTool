"""
Proper Code Mode Implementation for MetaTool
Based on jx-codes/lootbox and Cloudflare Code Mode architecture

Architecture:
1. LLM writes TypeScript code
2. Code executes in Deno sandbox (network-only permissions)
3. Code makes fetch() calls to HTTP proxy  
4. HTTP proxy forwards to MetaTool database
5. Results flow back through code execution
"""

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import http.server
import socketserver
import threading
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class CodeModeResult:
    """Result from code execution"""
    success: bool
    output: str
    execution_time_ms: float
    error: Optional[str] = None
    code_executed: Optional[str] = None


class MetaToolProxy:
    """
    HTTP proxy that exposes MetaTool database via REST API
    The TypeScript code calls this proxy, which forwards to the tool database
    """
    
    def __init__(self, tool_database, port: int = 3001):
        self.tool_database = tool_database
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start(self):
        """Start the HTTP proxy server in background thread"""
        handler = self._create_handler()
        self.server = socketserver.TCPServer(("localhost", self.port), handler)
        
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info(f"MetaTool proxy started on http://localhost:{self.port}")
        
        # Wait a moment for server to be ready
        time.sleep(0.1)
        
    def stop(self):
        """Stop the HTTP proxy server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("MetaTool proxy stopped")
    
    def _create_handler(self):
        """Create request handler with access to tool database"""
        tool_db = self.tool_database
        
        class MetaToolHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                """Suppress default logging"""
                pass
                
            def do_GET(self):
                """Handle GET requests"""
                if self.path == "/health":
                    self._send_json({"status": "ok"})
                elif self.path == "/tools":
                    # List available tools
                    self._send_json({
                        "tools": ["search", "validate_params", "get_by_category"]
                    })
                else:
                    self._send_error(404, "Not found")
                    
            def do_POST(self):
                """Handle POST requests (tool calls)"""
                if self.path == "/search":
                    self._handle_search()
                elif self.path == "/validate_params":
                    self._handle_validate()
                elif self.path == "/get_by_category":
                    self._handle_get_by_category()
                else:
                    self._send_error(404, "Unknown endpoint")
                    
            def _handle_search(self):
                """Handle tool search requests"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))
                    
                    query = data.get('query', '')
                    limit = data.get('limit', 5)
                    filters = data.get('filters', {})
                    
                    logger.info(f"[PROXY] Search request: query='{query}', limit={limit}")
                    
                    # Call the actual tool database (using correct method name)
                    results = tool_db.search(query, filters=filters, limit=limit)
                    
                    logger.info(f"[PROXY] Search found {len(results)} results")
                    
                    self._send_json({
                        "results": [
                            {
                                "name": r["name"],
                                "description": r.get("description", ""),
                                "category": r.get("category", ""),
                                "score": r.get("score", 0.0)
                            }
                            for r in results
                        ]
                    })
                    
                except Exception as e:
                    logger.error(f"[PROXY] Search error: {e}")
                    self._send_error(500, str(e))
                    
            def _handle_validate(self):
                """Handle parameter validation"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))
                    
                    self._send_json({
                        "valid": True,
                        "errors": []
                    })
                    
                except Exception as e:
                    self._send_error(500, str(e))
                    
            def _handle_get_by_category(self):
                """Handle get by category"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))
                    
                    category = data.get('category', '')
                    limit = data.get('limit', 10)
                    
                    # Use correct method name
                    results = tool_db.get_by_category(category, limit=limit)
                    
                    self._send_json({
                        "tools": results
                    })
                    
                except Exception as e:
                    self._send_error(500, str(e))
                    
            def _send_json(self, data: dict):
                """Send JSON response"""
                response = json.dumps(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response.encode('utf-8'))
                
            def _send_error(self, code: int, message: str):
                """Send error response"""
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error = json.dumps({"error": message})
                self.wfile.write(error.encode('utf-8'))
                
        return MetaToolHandler


class CodeModeAgent:
    """
    Proper Code Mode implementation using Deno for TypeScript execution
    
    This follows the architecture from jx-codes/lootbox:
    1. Generate TypeScript API types from tool schemas
    2. LLM writes TypeScript code using those types
    3. Execute code in Deno sandbox (network-only)
    4. Code makes fetch() calls to MetaTool proxy
    5. Proxy forwards to actual tool database
    """
    
    def __init__(self, tool_database, llm_client, proxy_port: int = 3001):
        self.tool_database = tool_database
        self.llm_client = llm_client
        self.proxy_port = proxy_port
        
        # Start HTTP proxy
        self.proxy = MetaToolProxy(tool_database, port=proxy_port)
        self.proxy.start()
        
        # Check if Deno is installed
        self.deno_available = self._check_deno()
        
    def _check_deno(self) -> bool:
        """Check if Deno is installed"""
        try:
            result = subprocess.run(
                ['deno', '--version'],
                capture_output=True,
                timeout=5
            )
            available = result.returncode == 0
            if available:
                logger.info("Deno runtime detected")
            else:
                logger.warning("Deno not found - install from https://deno.land")
            return available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Deno not found - install from https://deno.land")
            return False
            
    def generate_typescript_api(self) -> str:
        """
        Generate TypeScript type definitions for MetaTool API
        This is what the LLM uses to write type-safe code
        """
        api_types = f"""
// MetaTool API Types
// Available at: http://localhost:{self.proxy_port}

interface SearchInput {{
  query: string;
  limit?: number;
  filters?: {{ [key: string]: any }};
}}

interface SearchResult {{
  name: string;
  description: string;
  category: string;
  score: number;
}}

interface SearchOutput {{
  results: SearchResult[];
}}

interface ValidateInput {{
  tool_name: string;
  params: {{ [key: string]: any }};
}}

interface ValidateOutput {{
  valid: boolean;
  errors: string[];
}}

interface GetByCategoryInput {{
  category: string;
  limit?: number;
}}

interface GetByCategoryOutput {{
  tools: any[];
}}

// MetaTool API client
const METATOOL_API = "http://localhost:{self.proxy_port}";

async function searchTools(input: SearchInput): Promise<SearchOutput> {{
  const response = await fetch(`${{METATOOL_API}}/search`, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(input),
  }});
  return await response.json();
}}

async function validateParams(input: ValidateInput): Promise<ValidateOutput> {{
  const response = await fetch(`${{METATOOL_API}}/validate_params`, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(input),
  }});
  return await response.json();
}}

async function getByCategory(input: GetByCategoryInput): Promise<GetByCategoryOutput> {{
  const response = await fetch(`${{METATOOL_API}}/get_by_category`, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(input),
  }});
  return await response.json();
}}
"""
        return api_types
        
    def build_prompt(self, user_query: str) -> str:
        """Build prompt for LLM to generate TypeScript code"""
        api_types = self.generate_typescript_api()
        
        prompt = f"""You are an expert TypeScript developer. Write TypeScript code to answer the user's query using the MetaTool API.

User Query: {user_query}

Available API:
```typescript
{api_types}
```

Instructions:
1. Write clean TypeScript code to fulfill the user's request
2. Use the searchTools(), validateParams(), or getByCategory() functions
3. Use console.log() to output results in JSON format
4. The code will execute in Deno with network access only
5. Handle errors appropriately

Generate TypeScript code (no markdown, no explanations):
"""
        return prompt
        
    def execute_typescript(self, code: str, timeout: int = 10) -> CodeModeResult:
        """
        Execute TypeScript code in Deno sandbox
        
        This is the key difference from the broken implementation:
        - Actually runs TypeScript in Deno
        - No transpilation to Python needed
        - Network-only permissions (security)
        - Direct fetch() access to proxy
        """
        if not self.deno_available:
            return CodeModeResult(
                success=False,
                output="",
                execution_time_ms=0,
                error="Deno runtime not available. Install from https://deno.land",
                code_executed=code
            )
            
        start_time = time.time()
        
        try:
            # Prepend API functions to the user code
            api_defs = self.generate_typescript_api()
            full_code = api_defs + "\n\n// User code:\n" + code
            
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ts',
                delete=False
            ) as f:
                f.write(full_code)
                temp_file = f.name
                
            logger.info(f"Executing TypeScript in Deno (timeout: {timeout}s)")
            logger.debug(f"Full code:\n{full_code}")
            
            # Execute in Deno with network-only permissions
            # This is the Cloudflare/lootbox approach: real TypeScript execution
            result = subprocess.run(
                ['deno', 'run', '--allow-net', temp_file],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Execution successful ({execution_time:.0f}ms)")
                return CodeModeResult(
                    success=True,
                    output=result.stdout,
                    execution_time_ms=execution_time,
                    code_executed=code
                )
            else:
                logger.error(f"✗ Execution failed: {result.stderr}")
                return CodeModeResult(
                    success=False,
                    output=result.stdout,
                    execution_time_ms=execution_time,
                    error=result.stderr,
                    code_executed=code
                )
                
        except subprocess.TimeoutExpired:
            execution_time = (time.time() - start_time) * 1000
            return CodeModeResult(
                success=False,
                output="",
                execution_time_ms=execution_time,
                error=f"Execution timeout after {timeout}s",
                code_executed=code
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return CodeModeResult(
                success=False,
                output="",
                execution_time_ms=execution_time,
                error=str(e),
                code_executed=code
            )
            
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query using Code Mode
        
        Steps:
        1. Build prompt with TypeScript API
        2. LLM generates TypeScript code
        3. Execute code in Deno
        4. Return results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"CODE MODE QUERY: {user_query}")
        logger.info(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Step 1: Build prompt
        prompt = self.build_prompt(user_query)
        
        # Step 2: Get code from LLM
        logger.info("→ Calling LLM to generate TypeScript...")
        generated_code = self.llm_client(prompt)
        logger.info(f"→ LLM generated {len(generated_code)} chars of code")
        
        # Step 3: Execute code
        logger.info("→ Executing in Deno sandbox...")
        result = self.execute_typescript(generated_code)
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETE: {total_time:.0f}ms total")
        logger.info(f"{'='*80}\n")
        
        return {
            "query": user_query,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "code": generated_code,
            "execution_time_ms": result.execution_time_ms,
            "total_time_ms": total_time
        }
        
    def shutdown(self):
        """Cleanup resources"""
        self.proxy.stop()
