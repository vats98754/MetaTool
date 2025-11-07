"""
Detailed Logging and Debugging for Code Mode Agent

This adds comprehensive logging to trace data flow through:
1. TypeScript API generation
2. LLM code generation
3. Sandbox execution
4. RPC binding calls
5. Tool database queries
"""

import logging
import json
import sys
from pathlib import Path

# Create detailed logger
def setup_detailed_logger(log_file: str = "code_mode_debug.log"):
    """Setup detailed debug logger with multiple handlers."""
    
    # Create logger
    logger = logging.getLogger("code_mode_debug")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler - detailed logs
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - important logs only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Create global debug logger
debug_logger = setup_detailed_logger()


def log_data_flow(stage: str, data: any, data_name: str = "data"):
    """Log data at each stage of processing."""
    debug_logger.info(f"\n{'='*80}")
    debug_logger.info(f"STAGE: {stage}")
    debug_logger.info(f"{'='*80}")
    
    if isinstance(data, str):
        if len(data) > 500:
            debug_logger.debug(f"{data_name} (length={len(data)}):")
            debug_logger.debug(f"  First 200 chars: {data[:200]}")
            debug_logger.debug(f"  Last 200 chars: {data[-200:]}")
        else:
            debug_logger.debug(f"{data_name}: {data}")
    elif isinstance(data, dict):
        debug_logger.debug(f"{data_name} (dict with {len(data)} keys):")
        debug_logger.debug(json.dumps(data, indent=2, default=str))
    elif isinstance(data, list):
        debug_logger.debug(f"{data_name} (list with {len(data)} items):")
        for i, item in enumerate(data[:3]):  # Show first 3 items
            debug_logger.debug(f"  [{i}]: {item}")
        if len(data) > 3:
            debug_logger.debug(f"  ... and {len(data) - 3} more items")
    else:
        debug_logger.debug(f"{data_name}: {data}")
    
    debug_logger.info(f"{'='*80}\n")


def log_function_call(func_name: str, *args, **kwargs):
    """Log function call with arguments."""
    debug_logger.debug(f"\n>>> CALLING: {func_name}")
    if args:
        debug_logger.debug(f"    args: {args}")
    if kwargs:
        debug_logger.debug(f"    kwargs: {kwargs}")


def log_function_return(func_name: str, result: any):
    """Log function return value."""
    debug_logger.debug(f"<<< RETURNING from {func_name}")
    if isinstance(result, str) and len(result) > 200:
        debug_logger.debug(f"    result (truncated): {result[:200]}...")
    else:
        debug_logger.debug(f"    result: {result}")


def log_error(stage: str, error: Exception):
    """Log error with full context."""
    debug_logger.error(f"\n{'!'*80}")
    debug_logger.error(f"ERROR in {stage}")
    debug_logger.error(f"{'!'*80}")
    debug_logger.error(f"Error type: {type(error).__name__}")
    debug_logger.error(f"Error message: {str(error)}")
    debug_logger.error(f"{'!'*80}\n")
    
    # Also log stack trace
    import traceback
    debug_logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Test logging
    log_data_flow("TEST", {"key": "value", "number": 123}, "test_data")
    log_function_call("test_function", "arg1", "arg2", kwarg1="value1")
    log_function_return("test_function", "result_value")
    log_error("TEST", ValueError("This is a test error"))
