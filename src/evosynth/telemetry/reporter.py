"""
EventReporter: fire-and-forget POSTs of engine events to the telemetry server.

Settings: [telemetry] reporter_url, probe_timeout, post_timeout.
Event types: step, progress, tier_state, metric, composite, switch, log
(see docs: dashboard/sse-protocol).
"""
import threading

import requests

from evosynth.config import get_settings


class EventReporter:
    def __init__(self, base_url=None, probe_timeout=None, post_timeout=None):
        t = get_settings().telemetry
        base = (base_url or t.reporter_url).rstrip("/")
        self.url = f"{base}/event"
        self.post_timeout = t.post_timeout if post_timeout is None else post_timeout
        self.enabled = False
        try:
            # Telemetry is only enabled if the API server is already running
            requests.get(f"{base}/docs", timeout=t.probe_timeout if probe_timeout is None else probe_timeout)
            self.enabled = True
        except Exception:
            pass

    def emit(self, event_type, payload):
        if not self.enabled:
            return

        def _send():
            try:
                requests.post(self.url, json={"type": event_type, "data": payload}, timeout=self.post_timeout)
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()
