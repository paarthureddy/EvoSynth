"""Launcher for the telemetry relay (the app lives in evosynth.telemetry.server)."""
from evosynth.telemetry.server import app, serve  # noqa: F401  (app re-exported for api_server.py)

if __name__ == "__main__":
    serve()
