"""
ml_routes.py

Defines the REST API endpoints for direct machine learning operations.
Includes endpoints for checking model health and directly submitting 
JSON payloads for churn prediction, bypassing the conversational interface.
"""

from fastapi import APIRouter, Request, HTTPException
import traceback

from src.predictor import ChurnPredictor
from src.utils import validate_input_data, format_prediction_response

router = APIRouter()

# Instantiate the ML predictor at startup to avoid loading overhead per request
predictor = ChurnPredictor()

try:
    predictor.load()
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")


@router.get("/")
async def health_check():
    """
    Health check endpoint to verify API uptime and model status.
    Returns the current readiness state of the Churn Predictor.
    """
    return {
        "status": "healthy",
        "model_loaded": predictor.is_trained
    }


@router.post("/predict")
async def predict(request: Request):
    """
    Direct prediction endpoint.
    Accepts a raw JSON payload of customer data, validates it, 
    runs the churn prediction model, and returns the formatted results.
    """
    try:
        data = await request.json()
        
        # Ensure a payload was actually received
        if not data:
            raise HTTPException(status_code=400, detail="No JSON data provided")
        
        # Validate data structure and predict
        validate_input_data(data)
        result = predictor.predict_single(data)
        response = format_prediction_response(result)
        
        return response
        
    except ValueError as e:
        # Catch validation errors specifically as Bad Request (400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unforeseen errors as Internal Server Error (500)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")