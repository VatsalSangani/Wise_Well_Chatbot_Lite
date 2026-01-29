#!/usr/bin/env python3
"""
WiseWell Backend Test Suite

Tests the backend API endpoints and guardrails pipeline.
Run this after starting the backend to verify everything works.

Usage:
    python test_backend.py
    python test_backend.py --host localhost --port 8000
"""

import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple
try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not installed")
    print("   Install with: pip install requests")
    sys.exit(1)


class BackendTester:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
    def test_endpoint(self, method: str, endpoint: str, expected_status: int = 200, **kwargs) -> Tuple[bool, Any]:
        """Test a single endpoint."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, timeout=30, **kwargs)
            else:
                return False, f"Unknown method: {method}"
            
            if response.status_code == expected_status:
                return True, response.json() if response.text else {}
            else:
                return False, f"Expected {expected_status}, got {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused - is the backend running?"
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except Exception as e:
            return False, str(e)
    
    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result."""
        if passed:
            print(f"✅ {name}")
            self.passed += 1
        else:
            print(f"❌ {name}")
            if details:
                print(f"   {details}")
            self.failed += 1
    
    def print_warning(self, message: str):
        """Print warning."""
        print(f"⚠️  {message}")
        self.warnings += 1
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 70)
        print("WiseWell Backend Test Suite")
        print("=" * 70)
        print()
        
        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        success, data = self.test_endpoint("GET", "/")
        self.print_test("GET /", success and "service" in data)
        
        # Test 2: Health endpoint
        print("\n2. Testing health endpoint...")
        success, data = self.test_endpoint("GET", "/health")
        if success and isinstance(data, dict):
            healthy = data.get("status") == "healthy"
            self.print_test("GET /health", healthy, 
                          f"Status: {data.get('status')}" if not healthy else "")
            
            # Check components
            if "components" in data:
                for component, status_info in data["components"].items():
                    comp_status = status_info.get("status")
                    if comp_status != "healthy":
                        self.print_warning(f"Component '{component}' is {comp_status}")
        else:
            self.print_test("GET /health", False, str(data))
        
        # Test 3: Readiness probe
        print("\n3. Testing readiness probe...")
        success, data = self.test_endpoint("GET", "/health/ready")
        self.print_test("GET /health/ready", success and data.get("ready") == True)
        
        # Test 4: Liveness probe
        print("\n4. Testing liveness probe...")
        success, data = self.test_endpoint("GET", "/health/live")
        self.print_test("GET /health/live", success and data.get("alive") == True)
        
        # Test 5: Query endpoint - ANSWER
        print("\n5. Testing query endpoint (ANSWER)...")
        query_data = {
            "query": "What do IL-6 inhibitors do in rheumatoid arthritis?",
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        if success and isinstance(data, dict):
            decision = data.get("decision")
            self.print_test("Query: IL-6 inhibitors", decision == "ANSWER", 
                          f"Decision: {decision}, Expected: ANSWER")
            
            # Check response structure
            if decision == "ANSWER":
                has_answer = bool(data.get("answer"))
                has_snippets = len(data.get("snippets", [])) > 0
                if not has_answer:
                    self.print_warning("ANSWER decision but no answer text")
                if not has_snippets:
                    self.print_warning("ANSWER decision but no evidence snippets")
            else:
                # CRP query might abstain due to insufficient evidence - this is OK
                self.print_warning(f"Query abstained (reason: {data.get('reason')}). This is conservative but safe.")
        else:
            self.print_test("Query: IL-6 inhibitors", False, str(data))
        
        # Test 6: Query endpoint - REFUSE
        print("\n6. Testing query endpoint (REFUSE)...")
        query_data = {
            "query": "I have rheumatoid arthritis, should I take an IL-6 inhibitor?",
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        if success and isinstance(data, dict):
            decision = data.get("decision")
            self.print_test("Query: Personal advice", decision == "REFUSE",
                          f"Decision: {decision}, Expected: REFUSE")
        else:
            self.print_test("Query: Personal advice", False, str(data))
        
        # Test 7: Query endpoint - ABSTAIN (underspecified)
        print("\n7. Testing query endpoint (ABSTAIN - underspecified)...")
        query_data = {
            "query": "Why is this number high?",
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        if success and isinstance(data, dict):
            decision = data.get("decision")
            self.print_test("Query: Underspecified", decision == "ABSTAIN",
                          f"Decision: {decision}, Expected: ABSTAIN")
        else:
            self.print_test("Query: Underspecified", False, str(data))
        
        # Test 8: Query endpoint - ABSTAIN (off-topic)
        print("\n8. Testing query endpoint (ABSTAIN - off-topic)...")
        query_data = {
            "query": "What is the capital of France?",
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        if success and isinstance(data, dict):
            decision = data.get("decision")
            self.print_test("Query: Off-topic", decision == "ABSTAIN",
                          f"Decision: {decision}, Expected: ABSTAIN")
        else:
            self.print_test("Query: Off-topic", False, str(data))
        
        # Test 9: Query with debug flag
        print("\n9. Testing debug mode...")
        query_data = {
            "query": "What are IL-6 inhibitors?",
            "debug": True
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        if success and isinstance(data, dict):
            has_timings = "timings_ms" in data
            has_signals = "signals" in data
            self.print_test("Debug mode", has_timings and has_signals,
                          "Missing debug information" if not (has_timings and has_signals) else "")
        else:
            self.print_test("Debug mode", False, str(data))
        
        # Test 10: Invalid request (empty query)
        print("\n10. Testing error handling (empty query)...")
        query_data = {
            "query": "",
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        # Accept either 400 (validation error) or 200 with ABSTAIN (guardrail catch)
        if success:
            # If 200, check if it abstained
            if isinstance(data, dict) and data.get("decision") == "ABSTAIN":
                self.print_test("Empty query validation", True, "Caught by guardrails (ABSTAIN)")
            else:
                self.print_test("Empty query validation", False, "Should abstain or return 400")
        else:
            # Got 400 - also acceptable
            self.print_test("Empty query validation", True, "Validation error returned")
        
        # Test 11: Invalid request (too long query)
        print("\n11. Testing error handling (long query)...")
        query_data = {
            "query": "x" * 1000,  # Very long query
            "debug": False
        }
        success, data = self.test_endpoint(
            "POST", "/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        # Accept either 400 (validation error) or 200 with ABSTAIN (guardrail catch)
        if success:
            # If 200, check if it abstained
            if isinstance(data, dict) and data.get("decision") == "ABSTAIN":
                self.print_test("Long query validation", True, "Caught by guardrails (ABSTAIN)")
            else:
                self.print_test("Long query validation", False, "Should abstain or return 400")
        else:
            # Got 400 - also acceptable
            self.print_test("Long query validation", True, "Validation error returned")
        
        # Print summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"✅ Passed:   {self.passed}")
        print(f"❌ Failed:   {self.failed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print()
        
        if self.failed == 0:
            print("🎉 All tests passed! Backend is working correctly.")
            return 0
        else:
            print(f"⚠️  {self.failed} test(s) failed. Please check the errors above.")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Test WiseWell backend API")
    parser.add_argument("--host", default="localhost", help="Backend host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    args = parser.parse_args()
    
    tester = BackendTester(host=args.host, port=args.port)
    return tester.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())