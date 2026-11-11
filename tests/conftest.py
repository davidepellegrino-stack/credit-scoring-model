"""Shared fixtures: put src/ on the path and build a small, fixed dataset."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def credit_df():
    """2,000 synthetic applicants with a fixed seed, shared across tests."""
    return generate(n=2_000, seed=7)
