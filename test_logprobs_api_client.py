# =============================================================================
# Logprobs API Client for Kaggle LLM Server
# Retrieves token-level log probabilities for uncertainty quantification
# =============================================================================

import math
import json
import requests
from typing import Optional
from pydantic import BaseModel, Field, computed_field
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
SEED = 42
REPETITION_PENALTY = 1.05

# OpenAI-compatible client
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
)


# =============================================================================
# Pydantic Models for Structured Logprobs Output
# =============================================================================
class TokenLogprob(BaseModel):
    """Single token with its logprob and probability."""
    token: str
    logprob: float
    probability: float = Field(description="exp(logprob) - the actual probability")
    
    @computed_field
    @property
    def confidence_percent(self) -> str:
        """Return confidence as percentage string."""
        return f"{self.probability * 100:.1f}%"
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if token probability exceeds threshold."""
        return self.probability >= threshold
    
    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if token probability is below threshold (uncertain)."""
        return self.probability < threshold
    
    @classmethod
    def from_api(cls, token: str, logprob: float) -> "TokenLogprob":
        return cls(
            token=token,
            logprob=logprob,
            probability=math.exp(logprob)
        )


class LogprobsResponse(BaseModel):
    """Full response with token-level confidence data."""
    content: str
    tokens: list[TokenLogprob] = Field(default_factory=list)
    
    def mean_confidence(self) -> float:
        """Average probability across all tokens."""
        if not self.tokens:
            return 0.0
        return sum(t.probability for t in self.tokens) / len(self.tokens)
    
    def min_confidence_token(self) -> Optional[TokenLogprob]:
        """Find the least confident token in the response."""
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda t: t.probability)
    
    def low_confidence_tokens(self, threshold: float = 0.3) -> list[TokenLogprob]:
        """Get all tokens below confidence threshold."""
        return [t for t in self.tokens if t.is_low_confidence(threshold)]
    
    def confidence_summary(self) -> dict:
        """Summary statistics of token confidences."""
        if not self.tokens:
            return {
                "total_tokens": 0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "low_confidence_count": 0,
                "warning": "No logprobs returned - server may need --logits_all flag",
            }
        
        probs = [t.probability for t in self.tokens]
        return {
            "total_tokens": len(self.tokens),
            "mean_confidence": self.mean_confidence(),
            "min_confidence": min(probs),
            "max_confidence": max(probs),
            "low_confidence_count": len(self.low_confidence_tokens()),
        }


# =============================================================================
# Core Logprobs API Functions
# =============================================================================

def chat_with_logprobs(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    seed: int = SEED,
    repetition_penalty: float = REPETITION_PENALTY,
) -> LogprobsResponse:
    """
    Send a chat completion request and return logprobs for each token.
    
    Args:
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        LogprobsResponse with content and token-level probabilities
    """
    response = client.chat.completions.create(
        model="qwen3-coder",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={
            "top_k": top_k,
            "seed": seed,
            "repetition_penalty": repetition_penalty,
        },
        logprobs=True,
        top_logprobs=1,  # Required by llama-cpp-python to return logprobs
    )
    
    content = response.choices[0].message.content or ""
    tokens_data = []
    
    # Parse logprobs from response
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        for token_info in response.choices[0].logprobs.content:
            tokens_data.append(
                TokenLogprob.from_api(token_info.token, token_info.logprob)
            )
    
    return LogprobsResponse(content=content, tokens=tokens_data)


def chat_with_logprobs_stream(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    seed: int = SEED,
    repetition_penalty: float = REPETITION_PENALTY,
) -> LogprobsResponse:
    """
    Stream chat completion with real-time logprobs display.
    
    During streaming: prints each token with probability, e.g., "can(95.2%)"
    After streaming: returns LogprobsResponse with pure content and all token data.
    
    Args:
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        LogprobsResponse with content and token-level probabilities
    """
    import sys
    
    payload = {
        "model": "qwen3-coder",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "repetition_penalty": repetition_penalty,
        "logprobs": True,
        "top_logprobs": 1,
        "stream": True,
    }
    
    # Color codes for inline display
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    tokens_data = []
    content_parts = []
    
    print(f"\n{BOLD}🔄 Streaming with Logprobs:{RESET}\n")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if not line:
            continue
        
        line_str = line.decode("utf-8")
        if not line_str.startswith("data: "):
            continue
        
        data_str = line_str[6:]  # Remove "data: " prefix
        if data_str.strip() == "[DONE]":
            break
        
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        
        # Extract delta content and logprobs
        choices = chunk.get("choices", [])
        if not choices:
            continue
        
        choice = choices[0]
        delta = choice.get("delta", {})
        token_text = delta.get("content", "")
        
        if not token_text:
            continue
        

        logprobs_data = choice.get("logprobs")
        if logprobs_data and logprobs_data.get("content"):
            # Only use the first token_info (the selected token)
            token_info = logprobs_data["content"][0]
            token = token_info.get("token", token_text)
            logprob = token_info.get("logprob", 0.0)
            prob = math.exp(logprob)
            
       
            # Skip "Ghost Tokens": Placeholder chunks with extremely low logprobs (approx -25.16)
            GHOST_THRESHOLD = 1e-6 # 0.0001%
            if prob < GHOST_THRESHOLD and token_text.strip():
                continue
                
            # Skip "Chunk Duplicates": Skip if this token text exactly matches 
            # the last printed token text (arriving in a separate chunk)
            if content_parts and token_text == content_parts[-1]:
                # Update the logprob of the previous entry if this one is better
                if tokens_data and tokens_data[-1].token == token:
                    if logprob > tokens_data[-1].logprob:
                        tokens_data[-1].logprob = logprob
                        tokens_data[-1].probability = prob
                continue
            # --------------------------------------------------

            tokens_data.append(TokenLogprob.from_api(token, logprob))
            content_parts.append(token_text)
            
            # Choose color based on confidence
            if prob >= 0.8:
                color = GREEN
            elif prob >= 0.4:
                color = YELLOW
            else:
                color = RED
            
            # Print token with probability inline
            token_display = token.replace("\n", "\\n")
            print(f"{color}{token_display}{DIM}({prob*100:.1f}%){RESET}", end="", flush=True)
        else:
            # Fallback: print token without logprob if not already added
            if not content_parts or token_text != content_parts[-1]:
                print(f"{token_text}", end="", flush=True)
                content_parts.append(token_text)
    
    print("\n")  # End streaming output
    
    full_content = "".join(content_parts)
    
    # Print the final pure output
    print(f"{BOLD}📄 Final Output (Pure Text):{RESET}")
    print("-" * 40)
    print(full_content)
    print("-" * 40)
    
    return LogprobsResponse(content=full_content, tokens=tokens_data)


def chat_with_logprobs_raw(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_k: int = TOP_K,
    seed: int = SEED,
    repetition_penalty: float = REPETITION_PENALTY,
) -> dict:
    """
    Send a chat completion request and return raw JSON response with logprobs.
    Useful for debugging or when you need the full API response.
    """
    payload = {
        "model": "qwen3-coder",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "repetition_penalty": repetition_penalty,
        "logprobs": True,
        "top_logprobs": 1,  # Required by llama-cpp-python to return logprobs
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


# =============================================================================
# Display Helpers
# =============================================================================

def print_logprobs_colored(response: LogprobsResponse):
    """
    Print response with color-coded confidence levels.
    Uses ANSI escape codes for terminal colors.
    """
    print("\n" + "=" * 60)
    print("📊 LOGPROBS ANALYSIS")
    print("=" * 60)
    
    # Color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    print(f"\n{BOLD}Token-by-Token Confidence:{RESET}\n")
    
    for token_data in response.tokens:
        prob = token_data.probability
        
        # Choose color based on confidence
        if prob >= 0.8:
            color = GREEN
        elif prob >= 0.4:
            color = YELLOW
        else:
            color = RED
        
        token_display = repr(token_data.token)[1:-1]  # Remove quotes but keep escapes
        print(f"{color}{token_display}{RESET} ({prob*100:.1f}%)", end=" ")
    
    print("\n")


def print_logprobs_detailed(response: LogprobsResponse):
    """
    Print detailed logprobs information for each token.
    """
    print("\n" + "=" * 60)
    print("📊 DETAILED LOGPROBS")
    print("=" * 60)
    
    for i, token_data in enumerate(response.tokens):
        print(f"[{i+1}] {repr(token_data.token):20} → {token_data.probability*100:6.2f}% (logprob: {token_data.logprob:.4f})")


def print_confidence_summary(response: LogprobsResponse):
    """Print summary statistics of token confidences."""
    summary = response.confidence_summary()
    
    print("\n" + "=" * 60)
    print("📈 CONFIDENCE SUMMARY")
    print("=" * 60)
    print(f"  Total Tokens: {summary['total_tokens']}")
    print(f"  Mean Confidence: {summary['mean_confidence']*100:.1f}%")
    print(f"  Min Confidence: {summary['min_confidence']*100:.1f}%")
    print(f"  Max Confidence: {summary['max_confidence']*100:.1f}%")
    print(f"  Low Confidence Tokens (<30%): {summary['low_confidence_count']}")
    
    # Show lowest confidence token
    min_token = response.min_confidence_token()
    if min_token:
        print(f"\n  Least Confident Token: {repr(min_token.token)}")
        print(f"    Probability: {min_token.probability*100:.2f}%")


def print_low_confidence_tokens(response: LogprobsResponse, threshold: float = 0.3):
    """
    Print tokens with low confidence that may need verification.
    
    Note: Logprobs show confidence in the CHOSEN token.
    Low values may indicate temperature sampling picked a less-likely option.
    """
    low_conf = response.low_confidence_tokens(threshold)
    
    print("\n" + "=" * 60)
    print(f"⚠️ LOW CONFIDENCE TOKENS (<{threshold*100:.0f}%)")
    print("=" * 60)
    
    if not low_conf:
        print("  ✅ No low confidence tokens found!")
        return
    
    for token_data in low_conf:
        print(f"  • {repr(token_data.token):20} → {token_data.probability*100:.2f}%")


# =============================================================================
# Test Functions
# =============================================================================

def test_simple_logprobs():
    """Test: Simple question with logprobs."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Simple Logprobs")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    
    response = chat_with_logprobs(messages, max_tokens=50, temperature=0.1)
    
    print(f"\n📤 Content: {response.content}")
    print_logprobs_colored(response)
    print_confidence_summary(response)
    
    return response


def test_reasoning_logprobs():
    """Test: Reasoning task to see confidence variation."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Reasoning with Logprobs")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "Answer directly and concisely."
        },
        {
            "role": "user",
            "content": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer: Yes, No, or Cannot determine."
        }
    ]
    
    response = chat_with_logprobs(messages, max_tokens=100, temperature=0.1)
    
    print(f"\n📤 Content: {response.content}")
    print_logprobs_colored(response)
    print_confidence_summary(response)
    print_low_confidence_tokens(response)
    
    return response


def test_factual_logprobs():
    """Test: Factual question to see confidence on facts."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Factual Knowledge Logprobs")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ]
    
    response = chat_with_logprobs(messages, max_tokens=20, temperature=0.0)
    
    print(f"\n📤 Content: {response.content}")
    print_logprobs_detailed(response)
    print_confidence_summary(response)
    
    return response


def test_uncertainty_detection():
    """Test: Question designed to produce uncertainty."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Uncertainty Detection")
    print("=" * 60)
    
    messages = [
        {
            "role": "user",
            "content": "What will the weather be like tomorrow in Tokyo? Give a short prediction."
        }
    ]
    
    response = chat_with_logprobs(messages, max_tokens=100, temperature=0.7)
    
    print(f"\n📤 Content: {response.content}")
    print_confidence_summary(response)
    print_low_confidence_tokens(response, threshold=0.4)
    
    return response


def test_raw_logprobs():
    """Test: Get raw JSON response for inspection."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Raw Logprobs JSON")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Say hello."}
    ]
    
    raw_response = chat_with_logprobs_raw(messages, max_tokens=20)
    
    print("\n📤 Raw Response (truncated):")
    print(json.dumps(raw_response, indent=2)[:2000])
    
    return raw_response


def test_streaming_logprobs():
    """Test: Streaming with real-time logprobs display."""
    print("\n" + "=" * 60)
    print("🧪 TEST: Streaming Logprobs")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Explain why the sky is blue in 2-3 sentences."}
    ]
    
    response = chat_with_logprobs_stream(messages, max_tokens=100, temperature=0.3)
    
    print_confidence_summary(response)
    
    return response


# =============================================================================
# Difficulty-Graded Tests: Easy, Medium, Hard
# =============================================================================

def test_easy_math():
    """Test EASY: Simple arithmetic that should have very high confidence."""
    print("\n" + "=" * 60)
    print("🟢 TEST [EASY]: Simple Math (Streaming)")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a calculator. Answer with just the number, nothing else."
        },
        {
            "role": "user",
            "content": "What is 20 multiplied by 5, then divided by 4?"
        }
    ]
    
    response = chat_with_logprobs_stream(messages)
    
    print(f"   Expected: 25")
    print_confidence_summary(response)
    
    return response


def test_medium_reasoning():
    """Test MEDIUM: Multi-step word problem requiring some reasoning."""
    print("\n" + "=" * 60)
    print("🟡 TEST [MEDIUM]: Multi-step Word Problem (Streaming)")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "Solve the problem step by step, then give the final answer."
        },
        {
            "role": "user",
            "content": """A store has 3 shelves. Each shelf has 4 boxes. 
Each box contains 5 items. If 10 items are sold, how many items remain?
Think through this and give the final number."""
        }
    ]
    
    response = chat_with_logprobs_stream(messages)
    
    print(f"   Expected: 50 (3×4×5=60, 60-10=50)")
    print_confidence_summary(response)
    print_low_confidence_tokens(response, threshold=0.5)
    
    return response


def test_hard_logic_riddle():
    """Test HARD: Logic riddle requiring deductive reasoning."""
    print("\n" + "=" * 60)
    print("🔴 TEST [HARD]: Logic Riddle (Streaming)")
    print("=" * 60)
    
    messages = [
        {
            "role": "system",
            "content": "You are a logic expert. Reason carefully and give a definitive answer."
        },
        {
            "role": "user",
            "content": """Three people - Alice, Bob, and Carol - each have a different pet: 
a cat, a dog, or a fish.

Clues:
1. Alice does not have the fish.
2. Bob does not have the cat.
3. Carol does not have the dog.
4. The person with the cat is not Alice.

Who has which pet? Give the answer in format: Alice-[pet], Bob-[pet], Carol-[pet]"""
        }
    ]
    
    response = chat_with_logprobs_stream(messages)
    
    print(f"   Expected: Alice-dog, Bob-fish, Carol-cat")
    print_confidence_summary(response)
    print_low_confidence_tokens(response, threshold=0.5)
    
    return response


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "*" * 60)
    print("LOGPROBS API CLIENT - TOKEN CONFIDENCE ANALYSIS")
    print("*" * 60)
    print(f"\n📡 Base URL: {BASE_URL}")
    print(f"📊 Using logprobs for token confidence")
    
    tests = [
        ("🟢 EASY - Simple Math", test_easy_math),
        ("🟡 MEDIUM - Word Problem", test_medium_reasoning),
        ("🔴 HARD - Logic Riddle", test_hard_logic_riddle),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
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
    
    passed = sum(1 for r in results.values() if r)
    print(f"\n  Total: {passed}/{len(results)} passed")
    print("=" * 60)
