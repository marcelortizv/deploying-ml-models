# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Gradient Boosting Classifier using the default
hyperparameters in scikit-learn. The Goal of the model is to predict the Salary.

## Intended Use
This model should be used to predict the salary of a person based off a some 
attributes about it's financials.
## Training Data
Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; 
training process uses the 80% of data.
## Evaluation Data
Evaluation is done using 20% of this data.
## Metrics
The model was evaluated using Accuracy score. The value in laboratory is 
around 0.84.

## Ethical Considerations
This will drive to a model that may potentially discriminate people since contains 
information about the country, race and gender of the people.
## Caveats and Recommendations
This model was trained using a specific algorithm. It should be retrained testing other
algorithm or other hyperparemeters.
