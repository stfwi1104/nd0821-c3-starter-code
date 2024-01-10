from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import pytest
import pickle
import os

'''
@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
   
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    data_path = parent_path + "/data/"
    data_file = "census.csv"
    data = pd.read_csv(data_path+data_file)
    print("fsaf")
    return data


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape[0] == data.dropna().shape[0], "Dropping null changes shape."
    

def test_column_names(data):
    """ If your train and test set are assumed to have the same columnnames then this is a valid test. """
    train, test = train_test_split(data, test_size=0.20)
    assert set(train.columns)==set(test.columns)
'''

def test_classifier():
    """ If your model is an instance of DeciontreeCLassifier then this is a valid test. """
    model = pickle.load(open("./model/model.pkl",'rb'))
    assert isinstance(model, tree.DecisionTreeClassifier) 
