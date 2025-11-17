from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import socketio
import uvicorn
import pandas as pd
import asyncio

from model_utils import ModelPredictor


# ------------------------------------------------------------
# Initialize FastAPI + Socket.IO
# ------------------------------------------------------------
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

setattr(app, "sio", sio)


# ------------------------------------------------------------
# Load model + dataset
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_parquet("../data/arxiv_papers/train.parquet")
print(f"✓ Loaded dataset with {len(df)} rows")

print("Loading model...")
predictor = ModelPredictor("../best_model.pt")


# ------------------------------------------------------------
# Serve main HTML frontend
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
	with open("frontend/index.html", "r", encoding="utf-8") as f:
		return HTMLResponse(f.read())


# ------------------------------------------------------------
# Socket.IO Events
# ------------------------------------------------------------
@sio.event
async def connect(sid, environ):
	print(f"[+] Client connected: {sid}")


@sio.event
async def disconnect(sid):
	print(f"[-] Client disconnected: {sid}")


@sio.event
async def get_dataset_info(sid, data):
	info = {
		"total_samples": len(df),
		"columns": list(df.columns),
	}
	await sio.emit("dataset_info", info, room=sid)


@sio.event
async def get_sample(sid, data):
	index = data.get("index", 0)
	sample = predictor.get_sample_by_index(df, index)

	if sample:
		await sio.emit("sample_data", sample, room=sid)
	else:
		await sio.emit("error", {"message": "Sample not found"}, room=sid)


@sio.event
async def predict_sample(sid, data):
	index = data.get("index", 0)
	sample = predictor.get_sample_by_index(df, index)

	if not sample:
		await sio.emit("error", {"message": "Sample not found"}, room=sid)
		return

	text = sample["text"]
	predictions = predictor.predict_with_confidence(text)

	result = {
		"index": index,
		"predictions": predictions,
		"actual_label": sample["actual_label"],
	}

	await sio.emit("prediction_result", result, room=sid)


# ------------------------------------------------------------
# Attach Socket.IO to FastAPI
# ------------------------------------------------------------
app.mount("/", socketio.ASGIApp(sio, other_asgi_app=app))


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------
if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=6543)
