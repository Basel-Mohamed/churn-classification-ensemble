# Customer Churn Prediction
Base URL: ```http://127.0.0.1:5000``` (Local) / ```https://churn-classification-ensemble.vercel.app``` (Production)

This API provides endpoints for:
1- System Health: Monitoring the service status and model loading state.
2- Direct Churn Prediction: Submit raw customer data to get an instant risk assessment.

## 1. Health Check
Verifies if the API server is running and the machine learning model is loaded into memory.

Endpoint: ```/```

Method: ```GET```

### Response:

```JSON
{
  "status": "healthy",
  "model_loaded": true
}
```

## 2. Predict Customer Churn (Direct)
Directly predicts churn probability and risk level for a single customer based on structured JSON input.

Endpoint: ```/predict```

Method: ```POST``

Headers: ```Content-Type: application/json```

### Example Request

```JSON

{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "Senior_Citizen": 0,
    "Is_Married": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "Phone_Service": "No",
    "Dual": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "No",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "Yes",
    "Streaming_Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Monthly_Charges": 70.35,
    "Total_Charges": 844.2
}
```

### Success Response

```JSON

{
    "churn_prediction": "Yes",
    "churn_probability": 0.784,
    "risk_level": "High",
    "customerID": "7590-VHVEG"
}

```
