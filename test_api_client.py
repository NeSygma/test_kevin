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
USE_STREAMING = True  # Set to True to stream responses in real-time

# OpenAI-compatible client
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
)


# =============================================================================
# Streaming and Non-Streaming Completions
# =============================================================================

def stream_completion(messages: list, **kwargs) -> str:
    """
    Stream chat completion using OpenAI SDK.
    Returns the full content after streaming.
    """
    full_content = ""
    print("\n[RESULT]", flush=True)
    
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
            print(content, end="", flush=True)
    
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
    print(f"\n📤 Output:\n{content}")
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
    print("\n" + "*" * 60)
    print("KAGGLE LLM SERVER - COMPREHENSIVE API TEST")
    print("*" * 60)
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
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    passed_count = sum(1 for r in results.values() if r)
    total_count = len(results)
    print(f"\n  Total: {passed_count}/{total_count} passed")
    print("=" * 60)
