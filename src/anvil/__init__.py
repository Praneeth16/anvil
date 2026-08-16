"""ANVIL — self-mutating agent harness on Databricks."""

import os

# MLflow's trace exporter caches this setting when it is constructed, so the
# default must be established before any import can initialize MLflow tracing.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

__version__ = "0.2.0"
