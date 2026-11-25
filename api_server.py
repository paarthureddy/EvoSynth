"""
Root convenience wrapper: start the telemetry relay for the live dashboard.
"""
from evosynth.telemetry.server import serve

if __name__ == "__main__":
    serve()
