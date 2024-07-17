from data_loader import load_data, split_data
from preprocessor import TextPreprocessor
from model import SentimentModel

def train_and_evaluate_model():
    # Load and split data
    df = load_data('movie_reviews.csv')
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess data
    preprocessor = TextPreprocessor()
    X_train_vectorized = preprocessor.fit_transform(X_train)
    X_test_vectorized = preprocessor.transform(X_test)

    # Train model
    sentiment_model = SentimentModel()
    sentiment_model.train(X_train_vectorized, y_train)

    # Evaluate model
    accuracy, report = sentiment_model.evaluate(X_test_vectorized, y_test)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return preprocessor, sentiment_model

if __name__ == "__main__":
    train_and_evaluate_model()