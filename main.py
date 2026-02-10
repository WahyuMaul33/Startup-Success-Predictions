from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_engine import UMKMModelWrapper

app = FastAPI(
    title="UMKM Startup Success Predictor",
    description="API for predicting startup survival and growth based on investment and milestone data."
)

wrapper = UMKMModelWrapper("weights/gradient_boosting_model.pkl")

# Define the Input Schema (The Top 10 Drivers)
class StartupInput(BaseModel):
    age_startup_year: float
    age_last_milestone_year: float
    age_first_funding_year: float
    age_first_milestone_year: float
    funding_total_usd: float
    age_last_funding_year: float
    tier_relationships: int
    avg_participants: float
    milestones: int
    funding_rounds: int

@app.post("/predict")
async def predict_startup_status(data: StartupInput):
    """
    Accepts startup data and returns a survival status and probability 
    using the optimized XGBoost model.
    """
    try:
        user_data = data.model_dump()
        result = wrapper.predict(user_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "UMKM Success Predictor API is online", "model_version": "1.0.0"}