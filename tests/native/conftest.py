"""Keep argument-driven integration probes out of pytest collection.

These files remain executable commands with their historical CLI contracts;
they are not import-safe unit-test modules.
"""

collect_ignore = [
    "test_auto_acceleration.py",
    "test_coreml_resources.py",
    "test_packaged_independence.py",
    "test_preparation.py",
    "test_service.py",
    "test_staged_residency.py",
]
