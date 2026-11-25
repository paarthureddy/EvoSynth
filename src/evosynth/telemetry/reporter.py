"""
EventReporter: records engine events to the run's events.jsonl and, when the telemetry
server is running, sends them to it (fire-and-forget POSTs) for the live dashboard.

Settings: [telemetry] reporter_url, probe_timeout, post_timeout.
Event types: step, progress, tier_state, metric, composite, switch, log
(see docs: dashboard/sse-protocol). 'log' events (console text) go to the server only;
the console text itself is saved in run.log.

The active reporter is process-wide: get_reporter() / set_reporter(). A run context
(evosynth.telemetry.run_context.RunContext) installs one that also writes events.jsonl.
"""
import json
import threading
import time

import requests

from evosynth.config import get_settings

FILE_SKIP = {"log"}


def _json_default(obj):
    """numpy scalars/arrays and anything else json cannot encode."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class EventReporter:
    def __init__(self, base_url=None, probe_timeout=None, post_timeout=None, events_path=None):
        t = get_settings().telemetry
        base = (base_url or t.reporter_url).rstrip("/")
        self.url = f"{base}/event"
        self.post_timeout = t.post_timeout if post_timeout is None else post_timeout
        self._lock = threading.Lock()
        self._events = open(events_path, "a", encoding="utf-8") if events_path else None
        self.enabled = False
        try:
            # Live telemetry is only enabled if the API server is already running
            requests.get(f"{base}/docs", timeout=t.probe_timeout if probe_timeout is None else probe_timeout)
            self.enabled = True
        except Exception:
            pass

    def emit(self, event_type, payload):
        if self._events is not None and event_type not in FILE_SKIP:
            line = json.dumps({"t": round(time.time(), 3), "type": event_type, "data": payload},
                              default=_json_default, ensure_ascii=False)
            with self._lock:
                self._events.write(line + "\n")
                self._events.flush()
        if not self.enabled:
            return

        def _send():
            try:
                requests.post(self.url, json={"type": event_type, "data": payload}, timeout=self.post_timeout)
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()

    def close(self):
        if self._events is not None:
            with self._lock:
                self._events.close()
                self._events = None


_active = None


def get_reporter():
    """The process-wide reporter (created on first use: probes the server, no event file)."""
    global _active
    if _active is None:
        _active = EventReporter()
    return _active


def set_reporter(reporter):
    """Install `reporter` as the active one; returns the previous one (may be None)."""
    global _active
    previous, _active = _active, reporter
    return previous
