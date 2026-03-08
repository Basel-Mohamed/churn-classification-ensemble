"""
app.py

Main entry point for the e& Customer Churn Prediction API.
Initializes the FastAPI application, configures global settings like CORS, 
and registers the routing modules for both the Machine Learning endpoints 
and the conversational Chatbot agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ensure the root directory is in the path so 'src' module imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.ml_routes import router as ml_router
from api.chat_routes import router as chat_router

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="e& Customer Churn Prediction & Chatbot API",
    description="ML API and Conversational Agent for predicting customer churn",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing) to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the separated route modules
app.include_router(ml_router)
app.include_router(chat_router)

# Local development server execution
if __name__ == "__main__":
    import uvicorn
    # Run the app with hot-reloading enabled for development
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)