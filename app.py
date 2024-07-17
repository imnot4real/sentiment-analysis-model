from flask import Flask, render_template, request, jsonify
from data_loader import load_data, split_data
from preprocessor import TextPreprocessor
from model import SentimentModel

app = Flask(__name__)

# Load and preprocess data, train model
df = load_data('movie_reviews.csv')
X_train, X_test, y_train, y_test = split_data(df)

preprocessor = TextPreprocessor()
X_train_vectorized = preprocessor.fit_transform(X_train)

sentiment_model = SentimentModel()
sentiment_model.train(X_train_vectorized, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    review_vectorized = preprocessor.transform([review])
    prediction = sentiment_model.predict(review_vectorized)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)