"""
chat_routes.py

Handles the conversational AI endpoints using Cohere's LLM.
This module manages user chat sessions, extracts structured customer data 
from natural language inputs, and interfaces with the ML predictor once 
all required data points have been gathered.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import cohere
import json
import os
import traceback

from .ml_routes import predictor
from src.utils import format_prediction_response, validate_input_data

router = APIRouter()

# Initialize Cohere LLM client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# In-memory session store to track chat history and extracted data per user
sessions: Dict[str, Dict[str, Any]] = {}

# The complete list of features required by the ML model to make a prediction
REQUIRED_FIELDS = [
    "gender", "Senior_Citizen", "Is_Married", "Dependents",
    "tenure", "Phone_Service", "Dual", "Internet_Service", "Online_Security",
    "Online_Backup", "Device_Protection", "Tech_Support", "Streaming_TV",
    "Streaming_Movies", "Contract", "Paperless_Billing", "Payment_Method",
    "Monthly_Charges", "Total_Charges"
]

class ChatRequest(BaseModel):
    """Pydantic model for incoming chat messages."""
    message: str
    session_id: str

class ResetRequest(BaseModel):
    """Pydantic model for session reset requests."""
    session_id: str

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.
    Processes user messages, extracts required churn prediction features, 
    and triggers the ML model when the dataset is complete.
    """
    session_id = request.session_id
    user_message = request.message

    # Initialize a new session if one doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "extracted_data": {}
        }

    session = sessions[session_id]
    current_data = session["extracted_data"]
    
    # Identify which fields we still need to collect
    missing_fields = [f for f in REQUIRED_FIELDS if f not in current_data]

    # Dynamically construct the prompt for the LLM
    preamble = f"""You are a customer support agent gathering data to predict churn risk.
    Extract customer details from the user's message.
    
    Expected Fields: {REQUIRED_FIELDS}
    Currently Collected: {json.dumps(current_data)}
    Missing Fields: {missing_fields}

    CRITICAL INSTRUCTIONS FOR EXTRACTION:
    - "Senior_Citizen" must be 0 or 1.
    - "tenure" must be an integer.
    - "Monthly_Charges" and "Total_Charges" must be floats.
    - All other fields are strings.

    Return ONLY a valid JSON object with:
    1. "extracted_fields": Dictionary of NEW data found.
    2. "agent_reply": Your conversational response asking for 1 or 2 missing fields.
    """

    try:
        # Call Cohere LLM, forcing JSON output
        response = co.chat(
            message=user_message,
            preamble=preamble,
            chat_history=session["history"],
            response_format={"type": "json_object"}
        )

        # Parse LLM response
        llm_output = json.loads(response.text)
        new_extracted = llm_output.get("extracted_fields", {})
        agent_reply = llm_output.get("agent_reply", "Could you provide more details?")

        # Update session history and collected data
        session["history"].append({"role": "USER", "message": user_message})
        session["history"].append({"role": "CHATBOT", "message": agent_reply})
        session["extracted_data"].update(new_extracted)

        # Re-evaluate missing fields after latest extraction
        updated_missing = [f for f in REQUIRED_FIELDS if f not in session["extracted_data"]]

        # If all data is collected, proceed to prediction
        if not updated_missing:
            clean_data = session["extracted_data"].copy()
            
            # Type-cast numeric fields to satisfy ML model constraints
            try:
                if "Senior_Citizen" in clean_data:
                    clean_data["Senior_Citizen"] = int(clean_data["Senior_Citizen"])
                if "tenure" in clean_data:
                    clean_data["tenure"] = int(clean_data["tenure"])
                if "Monthly_Charges" in clean_data:
                    clean_data["Monthly_Charges"] = float(clean_data["Monthly_Charges"])
                if "Total_Charges" in clean_data:
                    clean_data["Total_Charges"] = float(clean_data["Total_Charges"])
            except ValueError as e:
                print(f"Data casting error: {e}")

            try:
                # Validate and predict
                validate_input_data(clean_data)
                
                prediction_result = predictor.predict_single(clean_data)
                formatted = format_prediction_response(prediction_result)
                
                print(f"\nDEBUG - ACTUAL FORMATTED RESPONSE: {formatted}\n")
                
                # Extract the nested 'data' dictionary to build the UI response
                prediction_data = formatted.get("data", {})
                
                prob_raw = prediction_data.get("churn_probability", prediction_data.get("probability", 0))
                prob = float(prob_raw) * 100 if prob_raw is not None else 0.0
                
                pred = prediction_data.get("churn_prediction", prediction_data.get("prediction", "Unknown"))
                risk = prediction_data.get("risk_level", prediction_data.get("risk", "Unknown"))
                
                customer_id = clean_data.get("customerID", "Unknown")
                
                # Format final text output for the user
                final_text = f"📊 Churn Prediction Results for Customer {customer_id}\n\n"
                final_text += f"🔮 Prediction: {pred}\n📈 Churn Probability: {prob:.1f}%\n⚠️ Risk Level: {risk}\n\n"
                
                if risk == "High":
                    final_text += "🚨 Alert: This customer has a HIGH risk of churning! Immediate retention actions recommended.\n\n"
                elif risk == "Medium":
                    final_text += "⚠️ Warning: This customer is at MEDIUM risk. Monitor closely and consider proactive outreach.\n\n"
                else:
                    final_text += "✅ This customer is currently at low risk.\n\n"
                    
                final_text += "💡 Recommendations:\n- Offer personalized retention incentives\n- Schedule a call from customer success team\n\nWould you like to check another customer?"

                return {
                    "status": "success",
                    "response": final_text,
                    "session_id": session_id,
                    "prediction_data": formatted,
                    "extracted_data": session["extracted_data"] # <--- ADDED: Syncs UI on final response
                }
                
            except ValueError as ve:
                # Handle cases where validation fails despite gathering all fields
                return {
                    "status": "error",
                    "response": f"⚠️ I collected all the data, but the model rejected it: {str(ve)}. Please reset the chat and try again.",
                    "session_id": session_id
                }

        # If data is still missing, return the LLM's follow-up question AND the extracted data
        return {
            "status": "success",
            "response": agent_reply,
            "session_id": session_id,
            "extracted_data": session["extracted_data"] # <--- ADDED: Progressively syncs UI during chat
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/chat/reset")
async def reset_chat(request: ResetRequest):
    """
    Clears the stored session data, allowing the user to start 
    a fresh churn prediction flow for a new customer.
    """
    session_id = request.session_id
    if session_id in sessions:
        sessions[session_id] = {
            "history": [],
            "extracted_data": {}
        }
    return {"status": "success", "message": "Session reset"}