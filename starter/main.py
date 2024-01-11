# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference 
import pickle
import pandas as pd
import json


app = FastAPI(
    title="API Salary Prediction",
    description="An API used for inference on the Census dataset.",
    version="1.0.0",
)


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Declare the data object with its components and their type.
class Inputfeatures(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country:  str


@app.get("/")
async def say_hello():
    return {"Welcome to the Project Application"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediciton/")
async def prediciton(data : Inputfeatures):

    model = pickle.load(open("model/model.pkl",'rb'))   
    encoder = pickle.load(open("model/encoder.pkl",'rb')) 
    lb = pickle.load(open("model/lb.pkl",'rb'))  
    
    #df = json.loads(data)
    #df = pd.DataFrame.from_dict(data)
    df = pd.DataFrame(data.dict(), index=[0])
    #print(df)
    #data = pd.read_json(data)
    X = process_data(df,cat_features,label=None,training=False,encoder=encoder,lb=lb)
   # pred = inference(model,data)

    print(df)
    return data