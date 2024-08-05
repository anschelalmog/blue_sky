import os
import sys
import unittest
import importlib.util
import traceback


def find_project_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, 'main.py')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find project root directory")
        current_dir = parent_dir


def load_tests_from_file(filename, tests_dir):
    # Get the full path to the file
    filepath = os.path.join(tests_dir, filename)

    # Load the module
    module_name = filename[:-3]  # Remove .py extension
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Return the test suite
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def run_all_tests():
    try:
        # Find the project root directory
        project_root = find_project_root()

        # Change to the project root directory
        os.chdir(project_root)

        # Add the project root to the Python path
        sys.path.insert(0, project_root)

        # Define the tests directory
        tests_dir = os.path.join(project_root, 'tests')

        # Ensure the tests directory exists
        if not os.path.exists(tests_dir):
            raise FileNotFoundError(f"Tests directory not found: {tests_dir}")

        # Create a test suite
        test_suite = unittest.TestSuite()

        # Get all test files in the tests directory
        # Get all test files in the tests directory
        test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]
        print(f"Found test files: {test_files}")

        if not test_files:
            print("No test files found. Ensure your test files start with 'test_' and end with '.py'")
            return False

        # Load tests from each file
        for test_file in test_files:
            file_path = os.path.join(tests_dir, test_file)
            print(f"\nInspecting file: {file_path}")
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(test_file[:-3], file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Inspect the module for test cases
                test_cases = [obj for name, obj in module.__dict__.items()
                              if isinstance(obj, type) and issubclass(obj, unittest.TestCase)]

                if not test_cases:
                    print(f"  No TestCase classes found in {test_file}")
                else:
                    for test_case in test_cases:
                        print(f"  Found TestCase: {test_case.__name__}")
                        test_methods = [method for method in dir(test_case) if method.startswith('test_')]
                        if not test_methods:
                            print(f"    No test methods found in {test_case.__name__}")
                        else:
                            print(f"    Test methods: {', '.join(test_methods)}")

                suite = unittest.defaultTestLoader.loadTestsFromModule(module)
                test_suite.addTest(suite)
                print(f"  Number of tests loaded from {test_file}: {suite.countTestCases()}")

            except Exception as e:
                print(f"  Error loading tests from {test_file}:")
                print(f"  {str(e)}")
                print("  Traceback:")
                traceback.print_exc()

        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)

        # Print summary
        print(f"\nTotal tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        # Return the result
        return result.wasSuccessful()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
