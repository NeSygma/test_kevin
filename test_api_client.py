# =============================================================================
# Comprehensive Test Client for Kaggle LLM Server
# Tests all API endpoints + Z3 Solver + Prolog examples
# =============================================================================

import requests
import json
from openai import OpenAI

# =============================================================================
# Configuration - Replace with your ngrok URL
# =============================================================================
BASE_URL = "https://curtate-unkeeled-sam.ngrok-free.dev"  # <-- Your ngrok URL

# Generation Defaults
MAX_TOKENS = 16384
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.05

# Display Settings
SHOW_THINKING = True  # Set to False to hide <think> content
USE_STREAMING = True  # Set to True to stream responses in real-time

# OpenAI-compatible client
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
)


# =============================================================================
# Helper: Parse thinking tags from response
# =============================================================================
import re

# Thinking tag patterns (different models use different tags)
THINK_OPEN_TAGS = ['<think>', '<thinking>']
THINK_CLOSE_TAGS = ['</think>', '</thinking>']

# For models that start thinking without explicit open tag (e.g., GLM with jinja template)
ASSUME_THINKING_MODE = True  # Set to True for reasoning models that may omit opening tag


def parse_thinking_response(content: str) -> tuple[str, str]:
    """
    Parse model response to separate thinking from output.
    Supports both <think> and <thinking> tags.
    
    Returns:
        (thinking, output) - thinking is empty string if no thinking tags found
    """
    # Try both patterns
    think_pattern = r'<think(?:ing)?>(.*?)</think(?:ing)?>'
    thinking_matches = re.findall(think_pattern, content, re.DOTALL)
    thinking = '\n'.join(match.strip() for match in thinking_matches)
    output = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
    return thinking, output


def print_response(content: str):
    """Print response with thinking separated if present."""
    thinking, output = parse_thinking_response(content)
    
    if thinking and SHOW_THINKING:
        print(f"\n💭 Thinking:\n{'-' * 40}")
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
        print(f"{'-' * 40}")
    
    print(f"\n📤 Output:")
    print(output)


def _is_think_open(content: str) -> bool:
    """Check if content contains a thinking open tag."""
    return any(tag in content for tag in THINK_OPEN_TAGS)


def _is_think_close(content: str) -> bool:
    """Check if content contains a thinking close tag."""
    return any(tag in content for tag in THINK_CLOSE_TAGS)


def _split_on_think_open(content: str) -> str:
    """Split content on opening thinking tag and return part after."""
    for tag in THINK_OPEN_TAGS:
        if tag in content:
            return content.split(tag)[-1]
    return content


def _split_on_think_close(content: str) -> tuple[str, str]:
    """Split content on closing thinking tag, return (before, after)."""
    for tag in THINK_CLOSE_TAGS:
        if tag in content:
            parts = content.split(tag)
            return parts[0], parts[-1] if len(parts) > 1 else ''
    return content, ''


def stream_completion(messages: list, **kwargs) -> str:
    """
    Stream chat completion using OpenAI SDK.
    Returns the full content after streaming.
    """
    full_content = ""
    in_thinking = ASSUME_THINKING_MODE  # Start in thinking mode if configured
    started_output = False
    printed_think_header = False
    
    # If assuming thinking mode, print header immediately
    if ASSUME_THINKING_MODE and SHOW_THINKING:
        print("\n[THINK]", flush=True)
        printed_think_header = True
    
    stream = client.chat.completions.create(
        model="qwen3-coder",
        messages=messages,
        max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
        temperature=kwargs.get("temperature", TEMPERATURE),
        top_p=kwargs.get("top_p", TOP_P),
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            
            # Handle thinking tags for display with block markers
            if _is_think_open(content):
                in_thinking = True
                if SHOW_THINKING and not printed_think_header:
                    print("\n[THINK]", flush=True)
                    printed_think_header = True
                # Print content after opening tag
                after_tag = _split_on_think_open(content)
                if after_tag and SHOW_THINKING:
                    print(after_tag, end="", flush=True)
            elif _is_think_close(content):
                # Print content before closing tag
                before_tag, after_tag = _split_on_think_close(content)
                if before_tag and SHOW_THINKING:
                    print(before_tag, end="", flush=True)
                if printed_think_header and SHOW_THINKING:
                    print("\n[/THINK]", flush=True)
                in_thinking = False
                # Start output section
                if not started_output:
                    print("\n[RESULT]", flush=True)
                    started_output = True
                # Print content after closing tag
                if after_tag:
                    print(after_tag, end="", flush=True)
            else:
                # Print token based on current state
                if in_thinking and SHOW_THINKING:
                    print(content, end="", flush=True)
                elif not in_thinking:
                    if not started_output and content.strip():
                        print("\n[RESULT]", flush=True)
                        started_output = True
                    print(content, end="", flush=True)
    
    if started_output:
        print("\n[/RESULT]", flush=True)
    
    return full_content


def non_stream_completion(messages: list, **kwargs) -> str:
    """Non-streaming chat completion using OpenAI SDK."""
    response = client.chat.completions.create(
        model="qwen3-coder",
        messages=messages,
        max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
        temperature=kwargs.get("temperature", TEMPERATURE),
        top_p=kwargs.get("top_p", TOP_P),
        stream=False,
    )
    content = response.choices[0].message.content
    print_response(content)
    return content


# =============================================================================
# API Endpoint Tests
# =============================================================================


def test_models_endpoint():
    """Test: GET /v1/models"""
    print("\n" + "=" * 60)
    print("📋 TEST: Models Endpoint (GET /v1/models)")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_docs_endpoint():
    """Test: GET /docs (Swagger UI)"""
    print("\n" + "=" * 60)
    print("📖 TEST: Docs Endpoint (GET /docs)")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/docs", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Response Length: {len(response.text)} bytes")
    return response.status_code == 200


def test_chat_completions():
    """Test: POST /v1/chat/completions"""
    print("\n" + "=" * 60)
    print("💬 TEST: Chat Completions (POST /v1/chat/completions)")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    
    try:
        if USE_STREAMING:
            stream_completion(messages)
        else:
            non_stream_completion(messages)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# Z3 Solver Example - Ask LLM to generate Z3 code
# =============================================================================

def test_z3_solver():
    """Test: Ask LLM to write Z3 solver code"""
    print("\n" + "=" * 60)
    print("🔢 TEST: Z3 Solver Code Generation")
    print("=" * 60)
    
    messages = [
        {
            "role": "system", 
            "content": """You are an expert in formal verification and constraint solving using Z3.
Write clean, executable Python code using z3-solver library.
Always include proper imports and make the code self-contained."""
        },
        {
            "role": "user", 
            "content": """Write a Z3 solver example that solves this logic puzzle:

There are 3 people: Alice, Bob, and Charlie.
- One of them is a knight (always tells truth), one is a knave (always lies), one is a spy (can lie or tell truth).
- Alice says: "I am not a spy."
- Bob says: "I am a knave."
- Charlie says: "Alice is the knight."

Who is the knight, knave, and spy?

Provide complete Z3 Python code to solve this."""
        }
    ]
    
    try:
        if USE_STREAMING:
            stream_completion(messages)
        else:
            non_stream_completion(messages)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# Prolog Example - Ask LLM to generate Prolog code
# =============================================================================

def test_prolog():
    """Test: Ask LLM to write Prolog code"""
    print("\n" + "=" * 60)
    print("🦉 TEST: Prolog Code Generation")
    print("=" * 60)
    
    messages = [
        {
            "role": "system", 
            "content": """You are an expert in Prolog and logic programming.
Write clean, executable Prolog code with proper comments.
Include example queries that demonstrate the code."""
        },
        {
            "role": "user", 
            "content": """Write a Prolog program that implements a family tree with the following:

1. Define facts for: parent(X, Y), male(X), female(X)
2. Include at least 3 generations of a family
3. Define rules for:
   - father(X, Y)
   - mother(X, Y)
   - grandparent(X, Y)
   - sibling(X, Y)
   - uncle(X, Y)
   - ancestor(X, Y)

Provide the complete Prolog code with example queries."""
        }
    ]
    
    try:
        if USE_STREAMING:
            stream_completion(messages)
        else:
            non_stream_completion(messages)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# Neuro-Symbolic Example - Combining Neural + Symbolic Reasoning
# =============================================================================

def test_neurosymbolic():
    """Test: Neuro-symbolic reasoning example"""
    print("\n" + "=" * 60)
    print("🧠 TEST: Neuro-Symbolic Reasoning")
    print("=" * 60)
    
    messages = [
        {
            "role": "system", 
            "content": """You are a neuro-symbolic AI assistant. For logical reasoning tasks:
1. First, identify the logical structure of the problem
2. Translate natural language to formal logic (predicates, rules)
3. Apply logical inference step by step
4. Provide the final answer with confidence

Be explicit about your reasoning process."""
        },
        {
            "role": "user", 
            "content": """Solve this logical reasoning problem step by step:

All mammals are warm-blooded.
All whales are mammals.
All dolphins are mammals.
No fish are warm-blooded.
Fluffy is either a whale or a fish.
Fluffy is warm-blooded.

Question: What is Fluffy?

First translate to formal logic, then reason step by step."""
        }
    ]
    
    try:
        if USE_STREAMING:
            stream_completion(messages)
        else:
            non_stream_completion(messages)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("KAGGLE LLM SERVER - COMPREHENSIVE API TEST")
    print("🚀" * 30)
    print(f"\n📡 Base URL: {BASE_URL}")
    
    results = {}
    
    # Test all endpoints
    tests = [
        ("Models (/v1/models)", test_models_endpoint),
        ("Docs (/docs)", test_docs_endpoint),
        ("Chat Completions", test_chat_completions),
        ("Z3 Solver", test_z3_solver),
        ("Prolog", test_prolog),
        ("Neuro-Symbolic", test_neurosymbolic),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for name, result in results.items():
        if result == "SKIP":
            status = "⏭️  SKIP"
        elif result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {status} - {name}")
    
    passed_count = sum(1 for r in results.values() if r is True)
    skip_count = sum(1 for r in results.values() if r == "SKIP")
    fail_count = sum(1 for r in results.values() if r is False)
    total_count = len(results)
    print(f"\n  Total: {passed_count} passed, {skip_count} skipped, {fail_count} failed")
    print("=" * 60)
