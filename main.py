from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import logging
import joblib
import time
import os
from pydantic import BaseModel, Field
from typing import Optional


# Initialize FastAPI app
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bot.servhub.fr", "http://bot.servhub.fr", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

model = joblib.load("models/model.pkl")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    index_path = os.path.join("static", "index.html")
    return FileResponse("./static/index.html")

@app.get("/sample/")
async def get_sample_data():
    try:
        logging.info("Received request to /sample/")
        df = pd.read_csv("well_predicted_full_dataset.csv")

        # Keep only the specified columns
        required_columns = ['PSH Flag Count', 'Min Packet Length', 'Bwd Packet Length Min', 'Day',
                            'Bwd Packets/s', 'Bwd Packet Length Std', 'URG Flag Count',
                            'Bwd Packet Length Max', 'Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV file: {missing_columns}")

        df = df[required_columns]

        # Replace NaN values with 0 (JSON-compatible)
        df = df.fillna(0)
        logging.debug(f"Sample data: {df.head()}")

        random_sample = df.sample(n=100)

        data = random_sample.head(100).to_dict(orient="records")  # Load first 100 rows for testing
        logging.info("Sample data successfully prepared.")
        return data
    except FileNotFoundError:
        logging.error("CSV file not found.")
        raise HTTPException(status_code=500, detail="CSV file not found.")
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty or invalid.")
        raise HTTPException(status_code=500, detail="CSV file is empty or invalid.")
    except Exception as e:
        logging.error(f"Error in /sample/ endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load sample data. {str(e)}")

class PredictionInput(BaseModel):
    PSH_Flag_Count: Optional[float] = Field(alias="PSH Flag Count", default=None)
    Min_Packet_Length: Optional[float] = Field(alias="Min Packet Length", default=None)
    Bwd_Packet_Length_Min: Optional[float] = Field(alias="Bwd Packet Length Min", default=None)
    Day: Optional[int] = None
    Bwd_Packets_per_s: Optional[float] = Field(alias="Bwd Packets/s", default=None)
    Bwd_Packet_Length_Std: Optional[float] = Field(alias="Bwd Packet Length Std", default=None)
    URG_Flag_Count: Optional[float] = Field(alias="URG Flag Count", default=None)
    Bwd_Packet_Length_Max: Optional[float] = Field(alias="Bwd Packet Length Max", default=None)

    class Config:
        allow_population_by_field_name = True

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Convert the input to a dictionary with the correct schema keys
        input_dict = input_data.dict(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        # Measure start time
        start_time = time.time()

        # Make predictions
        prediction = model.predict(input_df)

        # Calculate probabilities (if the model supports it)
        try:
            probabilities = model.predict_proba(input_df)
        except AttributeError:
            probabilities = None

        # Measure end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Return predictions, probabilities, and time
        return {
            "probabilities": probabilities.tolist() if probabilities is not None else "Not available",
            "prediction": prediction.tolist(),
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))