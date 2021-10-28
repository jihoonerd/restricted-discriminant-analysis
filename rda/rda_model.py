import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score


class RDA(BaseEstimator):
    """
    Restricted Discriminant Analysis (RDA) Estimator class.
    This inherits `BaseEstimator` class from `scikit-learn` to seamlessly integrate with other preprocessing module of it.
    RDA can model as LDA, QDA, nearest mean classifier, or mixture of them depending on hyperparameters, alpha, beta, and variance.
    """
    def __init__(self, alpha=0.0, beta=0.0, variance=0):
        """
        alpha, beta, and varaince decide the characteristics of RDA.

        Followings are remarks of RDA upon settings of hyperparameters:

        alpha = beta = 0:  QDA
        alpha=0, beta=1: LDA
        alpha=1, beta=0: nearest mean classifier
        """

        self.fitted = False

        self.alpha = alpha
        self.beta = beta
        self.variance = variance

        self.class_labels = []
        self.class_priors = {}
        self.class_means = {}

        self.rda_cov = {}
        
    def fit(self, X, y):

        self.class_labels = np.unique(y)

        S_i = {}
        pooled_covariance = 0
        # Calculate class priors, class means, and LDA pooled covariance matrix
        for i in self.class_labels:
            indices = np.where(y == i)[0]
            samples = X[indices, :]

            self.class_priors[i] = float(len(indices)) / len(y) # Class ratio
            self.class_means[i] = np.mean(samples, axis=0) # Mean vector of each features (N,NUM_FEATURES)

            S_i[i] = np.cov(samples, rowvar=0)

            pooled_covariance += S_i[i] * self.class_priors[i] # Equation (3) above.

        # Calculate RDA regularized covariance matricies for each class
        for i in self.class_labels:
            self.rda_cov[i] = self.alpha * self.variance * np.eye(X.shape[1]) + self.beta * pooled_covariance + (1 - self.alpha - self.beta) * S_i[i]
        self.fitted = True

    def predict(self, X):

        if not self.fitted:
            raise NameError('Fit model first')
        # Determine probability of each class given input vector

        predictions = []

        for i in range(len(X)):

            x = X[i]
            discriminant_score = {}
            for i in self.class_labels:
                # Divid the class delta calculation into 3 parts
                part1 = -0.5 * np.linalg.det(self.rda_cov[i])
                part2 = -0.5 * (x - self.class_means[i]).T @ np.linalg.pinv(self.rda_cov[i]) @ (x - self.class_means[i])
                part3 = np.log(self.class_priors[i])
                discriminant_score[i] = part1 + part2 + part3 # 5.19
            
            prediction = max(discriminant_score, key=discriminant_score.get)
            predictions.append(prediction)

        return predictions

    def score(self, X, y_true):

        y_pred = self.predict(X)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return f1_macro
