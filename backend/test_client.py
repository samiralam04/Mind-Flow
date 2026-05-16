import asyncio
import websockets
import json
import base64
import time
import numpy as np
import cv2
import glob
import os
import pandas as pd

async def test_session():
    # Create a dummy image (blue square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{b64}"

    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as ws:
        print("Connected to WS.")
        
        # Start Session
        await ws.send(json.dumps({
            "type": "START_SESSION",
            "metadata": {
                "participant_id": "test_user",
                "task_type": "automated_test",
                "difficulty": "medium"
            }
        }))
        
        resp = await ws.recv()
        print("Received:", resp)
        
        # Send 200 frames at ~30fps
        for i in range(200):
            await ws.send(json.dumps({
                "type": "FRAME",
                "data": data_uri,
                "timestamp": time.time()
            }))
            resp = await ws.recv()
            await asyncio.sleep(0.033)
            
        # Stop Session
        await ws.send(json.dumps({
            "type": "STOP_SESSION",
            "end_metadata": {
                "self_reported_load": 5
            }
        }))
        resp = await ws.recv()
        print("Received:", resp)
        
        await asyncio.sleep(1) # wait for batch writer to finish flush

    # Verify CSV
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "test_user")
    sessions = glob.glob(f"{dataset_dir}/ses_*")
    if not sessions:
        print("Error: No session directory found!")
        return
        
    latest_session = max(sessions, key=os.path.getctime)
    csv_path = os.path.join(latest_session, "features.csv")
    json_path = os.path.join(latest_session, "metadata.json")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Success! CSV has {len(df)} rows.")
        print(df.head(2))
    else:
        print(f"Error: CSV not found at {csv_path}")
        
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
            print(f"Success! Metadata: {meta['total_frames']} frames total.")

if __name__ == "__main__":
    asyncio.run(test_session())
