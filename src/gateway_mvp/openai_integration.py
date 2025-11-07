"""
Real OpenAI Integration for Code Mode

This replaces the mock LLM client with actual OpenAI API calls.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()


class OpenAICodeGenerator:
    """
    Real OpenAI client for Code Mode.
    
    Generates TypeScript code from user queries using GPT-4.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def __call__(self, prompt: str) -> str:
        """
        Generate TypeScript code from prompt.
        
        Args:
            prompt: The complete prompt including TypeScript API and user query
            
        Returns:
            Generated TypeScript code
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert TypeScript developer. Generate clean, "
                            "efficient TypeScript code to solve the user's request. "
                            "Use only the provided API. Output ONLY the TypeScript code, "
                            "no explanations or markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent code generation
                max_tokens=500,   # Reasonable limit for code snippets
            )
            
            # Extract the generated code
            code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```typescript"):
                code = code.split("```typescript", 1)[1]
                code = code.rsplit("```", 1)[0]
            elif code.startswith("```"):
                code = code.split("```", 1)[1]
                code = code.rsplit("```", 1)[0]
            
            return code.strip()
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def generate_with_metrics(self, prompt: str) -> dict:
        """
        Generate code and return detailed metrics.
        
        Returns dict with:
        - code: Generated TypeScript code
        - tokens_used: Total tokens consumed
        - prompt_tokens: Tokens in prompt
        - completion_tokens: Tokens in completion
        - model: Model used
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert TypeScript developer. Generate clean, "
                            "efficient TypeScript code to solve the user's request. "
                            "Use only the provided API. Output ONLY the TypeScript code, "
                            "no explanations or markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            # Extract code
            code = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if code.startswith("```typescript"):
                code = code.split("```typescript", 1)[1]
                code = code.rsplit("```", 1)[0]
            elif code.startswith("```"):
                code = code.split("```", 1)[1]
                code = code.rsplit("```", 1)[0]
            
            code = code.strip()
            
            # Extract metrics
            usage = response.usage
            
            return {
                "code": code,
                "tokens_used": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


def test_openai_integration():
    """Test the OpenAI integration."""
    print("Testing OpenAI Integration for Code Mode")
    print("="*80)
    
    # Create client
    llm = OpenAICodeGenerator(model="gpt-4")
    print(f"✓ OpenAI client initialized (model: gpt-4)")
    
    # Test prompt
    test_prompt = """You are an AI agent with access to the metaToolAPI API.

User Query: Find a tool to translate Spanish to English

Available TypeScript API:
```typescript
declare const metaToolAPI: {
  search: (input: {query: string, filters?: object, limit?: number}) => Promise<{results: any[]}>;
  validate_params: (input: {tool_name: string, params: object}) => Promise<{valid: boolean, errors: any[]}>;
  get_by_category: (input: {category: string, limit?: number}) => Promise<{tools: any[]}>;
};
```

Instructions:
1. Write TypeScript code to fulfill the user's request
2. Use the available API functions from the metaToolAPI object
3. Use console.log() to output results
4. The code will be executed in a secure sandbox
5. You have NO network access - only the API above

Generate TypeScript code (no explanations):
"""
    
    print("\nSending request to OpenAI GPT-4...")
    print("-"*80)
    
    # Generate code with metrics
    result = llm.generate_with_metrics(test_prompt)
    
    print("\n✓ Code generation successful!")
    print(f"\nModel: {result['model']}")
    print(f"Tokens Used: {result['tokens_used']}")
    print(f"  - Prompt: {result['prompt_tokens']}")
    print(f"  - Completion: {result['completion_tokens']}")
    print(f"Finish Reason: {result['finish_reason']}")
    
    print("\nGenerated TypeScript Code:")
    print("-"*80)
    print(result['code'])
    print("-"*80)
    
    return result


if __name__ == "__main__":
    test_openai_integration()
