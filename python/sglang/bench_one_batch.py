"""Back-compat shim. The implementation now lives in
``sglang.benchmark.one_batch``; this module preserves the
``python -m sglang.bench_one_batch`` entry point and the
``from sglang.bench_one_batch import ...`` imports.
"""

from sglang.benchmark.one_batch import *  # noqa: F401,F403
from sglang.benchmark.one_batch import cli_main

if __name__ == "__main__":
    cli_main()
