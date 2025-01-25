import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from predict import Predictor

app = FastAPI(title="LocationPredictor")
predictor = Predictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files supported")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:

            contents = await image.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        result = predictor.predict(tmp_path)

        tmp_path.unlink()

        return {
            "coordinates": result['coordinates'],
            "region": result['region'],
            "confidence": result['region_confidence']
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)