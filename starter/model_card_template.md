# Model Card

Model Card by Stefan


## Model Details
Classification
    - DecisionTreeClassifier from sklearn package
    - no hyperparamter tuning

## Intended Use
Prediction task is to determine whether a person makes over 50K a year.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income)
Data Extraction was done by Barry Becker from the 1994 Census database.  
Data for the project was split in 80% training data and 20% test/evaluation data

## Metrics
precision:  0.62
recall:     0.64
fbeta:      0.63

## Ethical Considerations
Data is public. Data has only a limit number of features. The correlation between sensitive data like sex, age and the salary (label) can be misleading.

## Caveats and Recommendations
Data is way to old and also to limited in behalf of the features to represent the actual social society...dont use it for real predictions; just for training.
