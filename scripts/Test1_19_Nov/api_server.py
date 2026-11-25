import os
import sys

# Ensure project root is in sys.path
_current = os.path.abspath(os.path.dirname(__file__))
while _current and not os.path.exists(os.path.join(_current, 'automl_engine')):
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    _current = _parent
_ROOT_DIR = _current
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


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

if __name__ == "__main__":
    import uvicorn
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

