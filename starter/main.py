# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
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
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status",example="Divorced")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=10)
    capital_loss: int = Field(..., alias="capital-loss", example=10)
    hours_per_week: int = Field(..., alias="hours-per-week", example=34)
    native_country: str = Field(...,alias="native-country", example="Germany")
   


@app.get("/")
async def say_hello():
    return {"Welcome to the Project Application"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/prediciton/")
async def prediciton(data : Inputfeatures):

    model = pickle.load(open("model/model.pkl",'rb'))   
    encoder = pickle.load(open("model/encoder.pkl",'rb')) 
    lb = pickle.load(open("model/lb.pkl",'rb'))  
  
    df = pd.DataFrame(data.dict(by_alias=True),  index=[0])
    print(df)
 
    X = process_data(df,cat_features,label=None,training=False,encoder=encoder,lb=lb)
    print(X[0])
    pred = inference(model,X[0])
    return data
