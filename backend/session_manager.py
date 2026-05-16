import asyncio
import aiofiles
import json
import os
import time
from datetime import datetime
from collections import namedtuple

# Ensure base dataset directory exists
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

CSV_HEADERS = [
    "timestamp", "frame_idx", "ear", "blink_count", "blink_duration_ms",
    "gaze_pitch", "gaze_yaw", "head_pitch", "head_yaw", "head_roll",
    "eyebrow_tension", "eye_openness", "fps", "light_intensity", "face_confidence"
]

class SessionManager:
    def __init__(self):
        # Maps session_id to session state
        self.sessions = {}

    async def start_session(self, metadata: dict) -> str:
        """Initialize a new recording session."""
        participant_id = metadata.get("participant_id", "anonymous")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"ses_{timestamp_str}"
        
        session_dir = os.path.join(DATASET_DIR, participant_id, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        csv_path = os.path.join(session_dir, "features.csv.tmp")
        meta_path = os.path.join(session_dir, "metadata.json")
        
        # Add backend generated metadata
        metadata["session_id"] = session_id
        metadata["start_time"] = datetime.now().isoformat()
        
        # Save metadata immediately
        async with aiofiles.open(meta_path, mode='w') as f:
            await f.write(json.dumps(metadata, indent=2))
            
        # Write CSV Header
        async with aiofiles.open(csv_path, mode='w') as f:
            await f.write(",".join(CSV_HEADERS) + "\n")
            
        queue = asyncio.Queue(maxsize=1000)
        
        # Start background writer task
        writer_task = asyncio.create_task(self._csv_writer_worker(session_id, queue, csv_path))
        
        self.sessions[session_id] = {
            "dir": session_dir,
            "csv_path": csv_path,
            "meta_path": meta_path,
            "queue": queue,
            "writer_task": writer_task,
            "metadata": metadata,
            "frame_idx": 0,
            "start_time": time.time(),
            "last_frame_time": None,
            "fps": 0.0,
            "blink_count": 0,
            "is_blinking": False,
            "blink_start_time": 0.0
        }
        
        print(f"Session {session_id} started.")
        return session_id
        
    async def log_frame(self, session_id: str, features: dict):
        """Enqueue frame features for writing."""
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        session["frame_idx"] += 1
        
        # Aggregator Logic: FPS, Blink Count, Blink Duration
        current_time = features.get("timestamp", time.time())
        
        # FPS Calculation
        if session["last_frame_time"] is not None:
            dt = current_time - session["last_frame_time"]
            if dt > 0:
                inst_fps = 1.0 / dt
                session["fps"] = (session["fps"] * 0.9) + (inst_fps * 0.1)
        else:
            session["fps"] = 0.0
        session["last_frame_time"] = current_time
        features["fps"] = session["fps"]
        
        # Blink Calculation
        is_blinking = features.get("is_blinking", False)
        was_blinking = session["is_blinking"]
        
        if is_blinking and not was_blinking:
            session["is_blinking"] = True
            session["blink_start_time"] = current_time
        elif not is_blinking and was_blinking:
            session["is_blinking"] = False
            session["blink_count"] += 1
            
        features["blink_count"] = session["blink_count"]
        if session["is_blinking"]:
            features["blink_duration_ms"] = (current_time - session["blink_start_time"]) * 1000.0
        else:
            features["blink_duration_ms"] = 0.0
        
        # Construct CSV row based on expected headers
        row_values = []
        for header in CSV_HEADERS:
            if header == "frame_idx":
                row_values.append(str(session["frame_idx"]))
            elif header == "timestamp":
                # Ensure we have a timestamp, default to now if missing
                ts = features.get(header, time.time())
                row_values.append(str(ts))
            else:
                val = features.get(header, "")
                if val is None or val == "NaN":
                    row_values.append("NaN")
                else:
                    # Format floats
                    row_values.append(f"{val:.6f}" if isinstance(val, float) else str(val))
                    
        row_str = ",".join(row_values) + "\n"
        
        try:
            # Don't block if queue is full (though unlikely with maxsize 1000 and 30fps)
            self.sessions[session_id]["queue"].put_nowait(row_str)
        except asyncio.QueueFull:
            print(f"Warning: Queue full for session {session_id}, dropping frame.")

    async def _csv_writer_worker(self, session_id: str, queue: asyncio.Queue, csv_path: str):
        """Background task that reads from the queue and writes to CSV in batches."""
        batch = []
        batch_size = 30  # Roughly 1 second of frames at 30fps
        
        try:
            while True:
                # Wait for a line to be added to the queue
                try:
                    # Timeout to ensure we flush even if no frames arrive
                    line = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if line is None:  # Sentinel value to stop
                        break
                    batch.append(line)
                    queue.task_done()
                except asyncio.TimeoutError:
                    pass # Just loop and check if we should flush
                    
                # Flush to disk if batch is full or we timed out with items
                if len(batch) >= batch_size or (len(batch) > 0 and queue.empty()):
                    async with aiofiles.open(csv_path, mode='a') as f:
                        await f.writelines(batch)
                    batch.clear()
                    
        except asyncio.CancelledError:
            pass
        finally:
            # Final flush on exit
            if batch:
                async with aiofiles.open(csv_path, mode='a') as f:
                    await f.writelines(batch)
            print(f"Writer worker for {session_id} finished.")

    async def stop_session(self, session_id: str, end_metadata: dict = None):
        """Stop a recording session and finalize files."""
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        
        # Put sentinel value to stop the worker
        await session["queue"].put(None)
        
        # Wait for worker to finish writing
        await session["writer_task"]
        
        # Rename tmp CSV to final CSV
        final_csv_path = session["csv_path"].replace(".tmp", "")
        os.rename(session["csv_path"], final_csv_path)
        
        # Update metadata
        metadata = session["metadata"]
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_frames"] = session["frame_idx"]
        
        if end_metadata:
            metadata.update(end_metadata)
            
        async with aiofiles.open(session["meta_path"], mode='w') as f:
            await f.write(json.dumps(metadata, indent=2))
            
        del self.sessions[session_id]
        print(f"Session {session_id} finalized.")

# Global instance
session_manager = SessionManager()
