import requests
import json

# feature input example for prediction 
featureinput =  { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
            }

response1 = requests.get('http://127.0.0.1:8000/')
response2 = requests.post('http://127.0.0.1:8000/prediciton/', data=json.dumps(featureinput))

print(response1.status_code)
print(response1.json())
print(response2.status_code)
print(response2.json())