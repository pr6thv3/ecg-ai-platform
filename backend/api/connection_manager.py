from fastapi import WebSocket
from typing import List

class ConnectionManager:
    """
    Manages active WebSocket connections for the real-time ECG stream.
    Supports multi-client broadcasting (e.g., multiple dashboards viewing the same patient).
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Sends data directly to a specific connected client."""
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """Broadcasts telemetry to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except RuntimeError:
                # Handle gracefully if the client disconnected during the broadcast
                self.disconnect(connection)
