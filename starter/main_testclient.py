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


response1 = client.get("/")
print(response1.status_code)
print(response1.json())


response2 = client.post('/prediciton/', data=json.dumps(featureinput))
print(response2.status_code)
print(response2.json())

#def test_api_locally_get_root():
 #   response1 = client.get("/")
  #  print(response1.status_code)
   # print(response1.json())
    
    
    #assert r.status_code == 200





#response1 = requests.get('http://127.0.0.1:8000/')
#response2 = requests.post('http://127.0.0.1:8000/prediciton/', data=json.dumps(featureinput))

#print(response1.status_code)
#print(response1.json())
#print(response2.status_code)
#print(response2.json())
