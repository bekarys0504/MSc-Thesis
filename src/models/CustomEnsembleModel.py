import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class CustomEnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # train each model
        for model in self.estimators:
            model.fit(X, y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
    
    
    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # get predictions with Demster-Shafer combination rule
        y_final = self.combine_using_Dempster_Schafer(X)

        return y_final

    def predict_proba(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # get predictions with Demster-Shafer combination rule
        y_pred_proba = self.combine_using_Dempster_Schafer(X, True)

        return y_pred_proba
    
    def predict_individual(self, X, proba):

        n_estimators = len(self.estimators)
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, n_estimators))


        for i, estimator in enumerate(self.estimators):
            if proba:
                y_pred[:, i] = estimator.predict_proba(X)[:, 1] #B
            else:
                y_pred[:, i] = estimator.predict(X) #C
        
        return y_pred

    def combine_using_Dempster_Schafer(self, X, proba=False):

        p_individual = self.predict_individual(X, proba=True) #A
        #print(p_individual)
        bpa0 = 1.0 - np.prod(p_individual, axis=1)
        bpa1 = 1 - np.prod(1 - p_individual, axis=1)
        belief = np.vstack([bpa0 / (1 - bpa0), bpa1 / (1 - bpa1)]).T #B
        y_final = np.argmax(belief, axis=1) #C
        if proba:
            return belief
        else:
            return y_final

    
