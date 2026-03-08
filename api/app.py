from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import ChurnPredictor
from src.utils import validate_input_data, format_prediction_response

app = FastAPI(
    title="Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
predictor = ChurnPredictor()

try:
    predictor.load()
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model - {str(e)}")


@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_trained
    }


@app.post("/predict")
async def predict(request: Request):
    """Single customer prediction endpoint"""
    try:
        # Get data from request
        data = await request.json()
        
        if not data:
            raise HTTPException(status_code=400, detail="No JSON data provided")
        
        # Validate input (using your existing utility)
        validate_input_data(data)
        
        # Make prediction
        result = predictor.predict_single(data)
        
        # Format response
        response = format_prediction_response(result)
        
        return response
        
    except ValueError as e:
        # Your validate_input_data likely raises ValueError on bad data
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Print full traceback for debugging in Vercel logs
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Local development runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)