# =============================================================================
# Logprobs API Client for Kaggle LLM Server
# Retrieves token-level log probabilities for uncertainty quantification
# =============================================================================

import math
import json
import requests
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

# =============================================================================
# Configuration - Replace with your ngrok URL
# =============================================================================
BASE_URL = "https://curtate-unkeeled-sam.ngrok-free.dev"  # <-- Your ngrok URL

# Generation Defaults
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8

# Logprobs Settings
TOP_LOGPROBS = 5  # Number of top alternative tokens to return (1-20)

# OpenAI-compatible client
client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key="not-needed",
)


# =============================================================================
# Data Classes for Structured Logprobs Output
# =============================================================================
@dataclass
class TokenLogprob:
    """Single token with its logprob and probability."""
    token: str
    logprob: float
    probability: float  # exp(logprob)
    
    @classmethod
    def from_logprob(cls, token: str, logprob: float) -> "TokenLogprob":
        return cls(
            token=token,
            logprob=logprob,
            probability=math.exp(logprob)
        )


@dataclass
class TokenWithAlternatives:
    """Token with its logprob and top alternative tokens."""
    token: str
    logprob: float
    probability: float
    alternatives: list[TokenLogprob] = field(default_factory=list)
    
    def confidence_percentile(self) -> str:
        """Return confidence as percentage string."""
        return f"{self.probability * 100:.1f}%"
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if token probability exceeds threshold."""
        return self.probability >= threshold
    
    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if token probability is below threshold (uncertain)."""
        return self.probability < threshold


@dataclass
class LogprobsResponse:
    """Full response with logprobs data."""
    content: str
    tokens: list[TokenWithAlternatives]
    
    def mean_confidence(self) -> float:
        """Average probability across all tokens."""
        if not self.tokens:
            return 0.0
        return sum(t.probability for t in self.tokens) / len(self.tokens)
    
    def min_confidence_token(self) -> Optional[TokenWithAlternatives]:
        """Find the least confident token in the response."""
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda t: t.probability)
    
    def low_confidence_tokens(self, threshold: float = 0.3) -> list[TokenWithAlternatives]:
        """Get all tokens below confidence threshold."""
        return [t for t in self.tokens if t.is_low_confidence(threshold)]
    
    def confidence_summary(self) -> dict:
        """Summary statistics of token confidences."""
        if not self.tokens:
            return {"error": "No tokens"}
        
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
    top_logprobs: int = TOP_LOGPROBS,
) -> LogprobsResponse:
    """
    Send a chat completion request and return logprobs for each token.
    
    Args:
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_logprobs: Number of top alternative tokens to return
    
    Returns:
        LogprobsResponse with content and token-level probabilities
    """
    response = client.chat.completions.create(
        model="qwen3-coder",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    
    content = response.choices[0].message.content or ""
    tokens_data = []
    
    # Parse logprobs from response
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        for token_info in response.choices[0].logprobs.content:
            # Main token
            main_token = TokenWithAlternatives(
                token=token_info.token,
                logprob=token_info.logprob,
                probability=math.exp(token_info.logprob),
                alternatives=[],
            )
            
            # Top alternative tokens
            if token_info.top_logprobs:
                for alt in token_info.top_logprobs:
                    if alt.token != token_info.token:  # Skip the main token
                        main_token.alternatives.append(
                            TokenLogprob.from_logprob(alt.token, alt.logprob)
                        )
            
            tokens_data.append(main_token)
    
    return LogprobsResponse(content=content, tokens=tokens_data)


def chat_with_logprobs_raw(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    top_logprobs: int = TOP_LOGPROBS,
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
        "logprobs": True,
        "top_logprobs": top_logprobs,
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


def print_logprobs_detailed(response: LogprobsResponse, show_alternatives: bool = True):
    """
    Print detailed logprobs information for each token.
    """
    print("\n" + "=" * 60)
    print("📊 DETAILED LOGPROBS")
    print("=" * 60)
    
    for i, token_data in enumerate(response.tokens):
        print(f"\n[{i+1}] Token: {repr(token_data.token)}")
        print(f"    Logprob: {token_data.logprob:.4f}")
        print(f"    Probability: {token_data.probability*100:.2f}%")
        
        if show_alternatives and token_data.alternatives:
            print("    Alternatives:")
            for alt in token_data.alternatives[:3]:  # Top 3 alternatives
                print(f"      - {repr(alt.token)}: {alt.probability*100:.2f}%")


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
        if min_token.alternatives:
            print(f"    Top Alternative: {repr(min_token.alternatives[0].token)}")


def print_low_confidence_tokens(response: LogprobsResponse, threshold: float = 0.3):
    """Print tokens with low confidence that may need verification."""
    low_conf = response.low_confidence_tokens(threshold)
    
    print("\n" + "=" * 60)
    print(f"⚠️ LOW CONFIDENCE TOKENS (<{threshold*100:.0f}%)")
    print("=" * 60)
    
    if not low_conf:
        print("  ✅ No low confidence tokens found!")
        return
    
    for token_data in low_conf:
        print(f"\n  Token: {repr(token_data.token)}")
        print(f"  Confidence: {token_data.probability*100:.2f}%")
        if token_data.alternatives:
            print(f"  Alternatives: {', '.join(repr(a.token) for a in token_data.alternatives[:3])}")


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
            "content": "Answer directly and concisely. No thinking tags."
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
    print_logprobs_detailed(response, show_alternatives=True)
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


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🔬" * 30)
    print("LOGPROBS API CLIENT - TOKEN CONFIDENCE ANALYSIS")
    print("🔬" * 30)
    print(f"\n📡 Base URL: {BASE_URL}")
    print(f"📊 Top Logprobs: {TOP_LOGPROBS}")
    
    tests = [
        ("Simple Logprobs", test_simple_logprobs),
        ("Factual Knowledge", test_factual_logprobs),
        ("Reasoning", test_reasoning_logprobs),
        ("Uncertainty Detection", test_uncertainty_detection),
        ("Raw JSON", test_raw_logprobs),
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
