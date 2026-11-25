"""
Telemetry relay: the engine POSTs events to /event, every connected dashboard receives
them as Server-Sent Events from /stream. In-memory, no history replay, no auth (local use).

Run: `python api_server.py` (host/port from [telemetry] in configs/evosynth.toml).
"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clients = set()


@app.post("/event")
async def receive_event(request: Request):
    data = await request.json()
    for client in list(clients):
        await client.put(data)
    return {"status": "ok"}


async def event_generator(queue: asyncio.Queue):
    try:
        while True:
            data = await queue.get()
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        pass


@app.get("/stream")
async def stream():
    queue = asyncio.Queue()
    clients.add(queue)

    async def cleanup():
        try:
            async for chunk in event_generator(queue):
                yield chunk
        finally:
            clients.discard(queue)

    return StreamingResponse(cleanup(), media_type="text/event-stream")


def serve(host=None, port=None):
    """Start the relay with uvicorn (blocking)."""
    import uvicorn
    from evosynth.config import get_settings
    t = get_settings().telemetry
    host = t.server_host if host is None else host
    port = t.server_port if port is None else port
    print(f"Starting API Server on http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
