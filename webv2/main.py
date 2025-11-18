from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import socketio
import uvicorn
import pandas as pd
from model_utils import ModelPredictor

class WebServer:
    def __init__(self):
        self.app = FastAPI()
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.df = pd.read_parquet("../data/arxiv_papers/train.parquet")
        self.predictor = ModelPredictor("../scibert_finetuned_model.pt")
        self._mount_routes()
        self._mount_socket_events()
        self.socket_app = socketio.ASGIApp(self.sio, other_asgi_app=self.app)

    def _mount_routes(self):
        self.app.mount("/static", StaticFiles(directory="frontend"), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            with open("frontend/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())

    def _mount_socket_events(self):
        @self.sio.event
        async def connect(sid, environ):
            pass

        @self.sio.event
        async def disconnect(sid):
            pass

        @self.sio.event
        async def get_dataset_info(sid, data):
            info = {
                "total_samples": len(self.df),
                "columns": list(self.df.columns)
            }
            await self.sio.emit("dataset_info", info, room=sid)

        @self.sio.event
        async def get_sample(sid, data):
            index = data.get("index", 0)
            sample = self.predictor.get_sample_by_index(self.df, index)
            if sample:
                await self.sio.emit("sample", sample, room=sid)
            else:
                await self.sio.emit("error", {"message": "Sample not found"}, room=sid)

        @self.sio.event
        async def predict_sample(sid, data):
            index = data.get("index", 0)
            sample = self.predictor.get_sample_by_index(self.df, index)

            if not sample:
                await self.sio.emit("error", {"message": "Sample not found"}, room=sid)
                return

            text = sample["text"]
            result = self.predictor.predict_with_embeddings(text)

            payload = {
                "index": index,
                "predictions": result["predictions"],
                "actual_label": sample["actual_label"],
                "sample_embedding": result["sample_embedding"],
                "prediction_embeddings": result["prediction_embeddings"]
            }

            await self.sio.emit("prediction_result", payload, room=sid)


if __name__ == "__main__":
    server = WebServer()
    uvicorn.run(server.socket_app, host="0.0.0.0", port=6543)
