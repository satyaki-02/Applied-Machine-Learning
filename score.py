from sklearn.base import BaseEstimator
import numpy as np
import sklearn

def score(text: str, model, threshold: float) -> tuple[bool, float]:
    
    # Convert text into a format the model can process
    # vectorized_text = np.array([text])  # Assumes model can handle raw text or requires a vectorizer
    
    # Get probability predictions
    probabilities = model.predict_proba([text])
    
    # Extract the probability of the positive class (assumed to be class 1)
    propensity = probabilities[0, 1]
    
    # Determine prediction based on threshold
    prediction = propensity >= threshold
    
    return bool(prediction), propensity