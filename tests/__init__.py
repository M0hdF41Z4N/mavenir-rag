from unittest import TestLoader, TextTestRunner

from coverage import Coverage


def run_tests():
    cov = Coverage()
    cov.start()
    loader = TestLoader()
    # Ref: https://docs.python.org/3/library/unittest.html#unittest.TestLoader.discover
    suite = loader.discover("tests", pattern="test_*.py")
    runner = TextTestRunner()
    runner.run(suite)
    cov.stop()
    cov.report(show_missing=True)  # This will print out the coverage report to the console
    cov.html_report(directory="python_coverage")  # Optional: Create an HTML report
