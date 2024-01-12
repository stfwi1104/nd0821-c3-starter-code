import requests
import json
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# feature input example for prediction 
featureinput =  { 'age':20,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"bachelors",
            'education-num':16,
            'marital-status':"Separated",
            'occupation':"Sales",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Male",
            'capital-gain':0,
            'capital-loss':0,
            'hours-per-week':10,
            'native-country':"United-States"
            }

'''
response1 = client.get("/")
print(response1.status_code)
print(response1.json())


response2 = client.post('/prediciton/', data=json.dumps(featureinput))
print(response2.status_code)
print(response2.json())
'''

# test statuscode on root
def test_api_locally_get_root():
    response = client.get("/")
    assert response.status_code == 200

# test content on root
def test_api_locally_get_root():
    response = client.get("/")
    assert response.json() is not None

# test statuscode on inference
def test_api_inference():
    response = client.post('/prediciton/', data=json.dumps(featureinput))
    assert response.status_code == 200

# test prediction on inference 
def test_api_inference_prediction():
    response = client.post('/prediciton/', data=json.dumps(featureinput))
    assert (response.json()["Prediction"] == '<50k') | (response.json() == (response.json()["Prediction"] == '>50k')) 

