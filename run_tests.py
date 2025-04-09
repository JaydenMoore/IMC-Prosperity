#!/usr/bin/env python3
import unittest
import argparse
import sys
import time
from test_trader import TestSQUIDINKStrategy
from market_simulator import run_market_scenarios
from sample_data import run_test_suite

def run_unit_tests():
    """Run unit tests using the unittest framework"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSQUIDINKStrategy)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

def measure_performance():
    """Measure algorithm performance using sample data"""
    from main import Trader
    from sample_data import SampleDataGenerator
    
    print("Measuring algorithm performance...")
    print("Running performance tests with 100 iterations each...")
    
    generator = SampleDataGenerator()
    test_cases = generator.generate_all_test_cases()
    trader = Trader()
    
    results = {}
    
    for name, state in test_cases.items():
        print(f"\nTesting scenario: {name}")
        
        # Run 100 iterations to measure average time
        times = []
        for _ in range(100):
            start_time = time.time()
            orders, conversions, trader_data = trader.run(state)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"  Average runtime: {avg_time:.2f}ms")
        print(f"  Maximum runtime: {max_time:.2f}ms")
        
        # Check if it's within the 900ms limit
        status = "✓ PASS" if max_time < 900 else "✗ FAIL"
        print(f"  Performance check: {status}")
        
        results[name] = {
            "avg_time": avg_time,
            "max_time": max_time,
            "pass": max_time < 900
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test the SQUID_INK trading algorithm')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--scenarios', action='store_true', help='Run market scenarios simulation')
    parser.add_argument('--performance', action='store_true', help='Measure algorithm performance')
    parser.add_argument('--sample', action='store_true', help='Run sample data tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no arguments specified, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 1
    
    # Track if any tests failed
    failed = False
    
    # Run unit tests
    if args.unit or args.all:
        print("\n=== Running Unit Tests ===\n")
        result = run_unit_tests()
        if result.failures or result.errors:
            failed = True
    
    # Run market scenarios
    if args.scenarios or args.all:
        print("\n=== Running Market Scenarios ===\n")
        run_market_scenarios()
    
    # Run performance tests
    if args.performance or args.all:
        print("\n=== Running Performance Tests ===\n")
        perf_results = measure_performance()
        if not all(result["pass"] for result in perf_results.values()):
            failed = True
    
    # Run sample data tests
    if args.sample or args.all:
        print("\n=== Running Sample Data Tests ===\n")
        run_test_suite()
    
    # Return non-zero exit code if any tests failed
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())