import requests

url = "http://127.0.0.1:5000/predict_api"
data = {
    "data": {
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322,
        "AveOccup": 2.5556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
}

# Send the POST request with JSON data
response = requests.post(url, json=data)

# Print the prediction response
print(response.json())
