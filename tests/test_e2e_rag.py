import os
import unittest


class E2ERagPipelineTest(unittest.TestCase):
    """End-to-end integration test for the RAG pipeline.

    Gated behind the RUN_E2E env var so the default `poetry run test` suite
    (unit tests) does not hit the network or require a live API server.

    To run: RUN_E2E=1 poetry run test
    Or directly: poetry run python -m tests.e2e.runner
    """

    @unittest.skipUnless(os.environ.get("RUN_E2E") == "1", "Set RUN_E2E=1 to run the E2E pipeline test")
    def test_full_pipeline_produces_reports(self) -> None:
        from tests.e2e.runner import main

        exit_code = main()
        self.assertEqual(exit_code, 0, f"runner.main() returned non-zero exit code: {exit_code}")

        from tests.e2e.config import REPORTS_DIR

        run_dirs = sorted(
            (p for p in REPORTS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        self.assertTrue(run_dirs, f"No run directory found under {REPORTS_DIR}")
        latest = run_dirs[-1]
        summary = latest / "summary.html"
        detailed = latest / "detailed.html"
        self.assertTrue(summary.exists() and summary.stat().st_size > 0, f"Missing/empty {summary}")
        self.assertTrue(
            detailed.exists() and detailed.stat().st_size > 0,
            f"Missing/empty {detailed}",
        )
