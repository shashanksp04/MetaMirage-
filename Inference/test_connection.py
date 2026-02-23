#!/usr/bin/env python3
"""
Quick test script to verify connection to the Qwen model server.
"""
import sys
import requests
from openai import OpenAI

# Configuration - update these to match your setup
API_BASE = "http://127.0.0.1:11434/v1"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print("=" * 60)
print("Testing Connection to Qwen Model Server")
print("=" * 60)
print(f"API Base: {API_BASE}")
print(f"Model: {MODEL_NAME}")
print()

# Test 1: Check if server is reachable
print("Test 1: Checking if server is reachable...")
try:
    response = requests.get(f"{API_BASE}/models", timeout=5)
    if response.status_code == 200:
        print("✅ Server is reachable!")
        models = response.json()
        print(f"   Available models: {models.get('data', [])}")
    else:
        print(f"❌ Server returned status code: {response.status_code}")
        print(f"   Response: {response.text}")
except requests.exceptions.ConnectionError:
    print("❌ Connection Error: Cannot reach server")
    print("   Possible issues:")
    print("   1. Server is not running")
    print("   2. Wrong port number (check if server is on port 11434)")
    print("   3. Wrong host (try 'localhost' instead of '127.0.0.1')")
    print()
    print("   To start the server, run:")
    print(f"   python -m vllm.entrypoints.openai.api_server --model {MODEL_NAME} --host 127.0.0.1 --port 11434 --tensor-parallel-size 1")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print()

# Test 2: Try to initialize OpenAI client
print("Test 2: Testing OpenAI client initialization...")
try:
    client = OpenAI(
        api_key="dummy-key",
        base_url=API_BASE
    )
    print("✅ Client initialized successfully")
except Exception as e:
    print(f"❌ Client initialization failed: {e}")
    sys.exit(1)

print()

# Test 3: Try a simple chat completion
print("Test 3: Testing chat completion...")
try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("✅ Chat completion successful!")
    print(f"   Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Chat completion failed: {e}")
    print()
    print("   Common issues:")
    print(f"   1. Model name '{MODEL_NAME}' doesn't match server model")
    print("   2. Server is still loading the model")
    print("   3. Out of memory (check server logs)")
    sys.exit(1)

print()
print("=" * 60)
print("✅ All tests passed! Connection is working.")
print("=" * 60)

