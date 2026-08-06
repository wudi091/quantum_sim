"""Upstream SimQN reference sources, excluded from the SeQUeNCe test suite."""


def load_tests(loader, tests, pattern):
    """Keep upstream reference scripts out of project unittest discovery."""

    return loader.suiteClass()
