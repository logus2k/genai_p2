# main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import socketio
import uvicorn
import asyncio
import pandas as pd
from model_utils import ModelPredictor


# Initialize model predictor
predictor = ModelPredictor("./best_model.pt")

# Load dataset
df = pd.read_parquet("./data/arxiv_papers/train.parquet")

# Initialize Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*")
app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.sio = sio

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(f.read())

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def get_sample(sid, data):
    """Get a specific sample from the dataset"""
    index = data.get('index', 0)
    sample = predictor.get_sample_by_index(df, index)
    
    if sample:
        await sio.emit('sample_data', sample, room=sid)
    else:
        await sio.emit('error', {'message': 'Sample not found'}, room=sid)

@sio.event
async def predict_sample(sid, data):
    """Get model prediction for a specific sample"""
    index = data.get('index', 0)
    sample = predictor.get_sample_by_index(df, index)
    
    if sample:
        # Get predictions
        predictions = predictor.predict_with_confidence(sample['text'])
        
        result = {
            'index': index,
            'predictions': predictions,
            'actual_label': sample['actual_label']
        }
        
        await sio.emit('prediction_result', result, room=sid)
    else:
        await sio.emit('error', {'message': 'Sample not found'}, room=sid)

@sio.event
async def get_dataset_info(sid, data):
    """Get dataset information"""
    info = {
        'total_samples': len(df),
        'columns': list(df.columns)
    }
    await sio.emit('dataset_info', info, room=sid)

@app.middleware("http")
async def add_socketio_middleware(request: Request, call_next):
    request.sio = sio
    response = await call_next(request)
    return response

# Mount socket.io
app.mount('/', socketio.ASGIApp(sio, other_asgi_app=app))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6543)
