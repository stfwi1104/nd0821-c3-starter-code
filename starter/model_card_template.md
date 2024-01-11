# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Classification
    - DecisionTreeClassifier from sklearn package
    - no hyperparamter tuning

## Intended Use
Prediction task is to determine whether a person makes over 50K a year.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income)
Data Extraction was done by Barry Becker from the 1994 Census database.  
Data was split in 80% training data and 20% test/evaluation data

## Metrics
precision:  0.62
recall:     0.64
fbeta:      0.63

## Ethical Considerations


## Caveats and Recommendations
Data is way to old to represent the actual social society...dont use it for real predictions
