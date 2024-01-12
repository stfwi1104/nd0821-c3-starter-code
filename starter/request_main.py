import requests
import json

# feature input example for prediction 
featureinput =  { 'age':20,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Masters",
            'education-num':16,
            'marital-status':"Separated",
            'occupation':"Sales",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Male",
            'capital-gain':0,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':"United-States"
            }



response1 = requests.get('https://udacityproject-e5aeb4bbcf6e.herokuapp.com/')
response2 = requests.post('https://udacityproject-e5aeb4bbcf6e.herokuapp.com/prediciton/', data=json.dumps(featureinput))

print(response1.status_code)
print(response1.json())
print(response2.status_code)
print(response2.json())
