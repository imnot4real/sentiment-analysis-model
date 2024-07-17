from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)

    def transform(self, X):
        return self.vectorizer.transform(X)