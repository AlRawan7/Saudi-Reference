from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from tabulate import tabulate
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json

# nltk.download('punkt')
# nltk.download('stopwords')

app = Flask(__name__, static_folder='static')

# Download NLTK resources
# nltk.download('stopwords')

# Load and preprocess data
data = pd.read_csv("DataAfterPreAndFeatures.csv")

# Impute missing values with empty strings
data['Tokens'].fillna('', inplace=True)

# Convert non-string data to strings
data['Tokens'] = data['Tokens'].astype(str)

# Build the Tf-Idf model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Tokens'])

@app.route('/')
def home():
    return render_template('page1.html')


@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')


@app.route('/page2.html')
def whereto():
    return render_template('page2.html')

@app.route('/page3', methods=['POST'])
def page3():
    # Retrieve user inputs from the form
    user_location = request.form['location']
    user_description = request.form['description']

    # Get user location
    geolocator = Nominatim(user_agent="place_recommendation")
    location = geolocator.geocode(user_location)
    user_latitude = request.form['latitude']
    user_longitude = request.form['longitude']

    # Preprocess user input
    def preprocess_text(text):
        text = text.lower()
        text = re.sub('[^A-Za-z]', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    user_tokens = preprocess_text(user_description)
    # Preprocess user input
    preprocessed_user_input = vectorizer.transform([user_tokens])
    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix, preprocessed_user_input)
    # Add similarity scores to the DataFrame
    data['Similarity'] = similarities.flatten()

    # Calculate distances from user location in kilometers
    data['Distance'] = data.apply(
        lambda row: geodesic((user_latitude, user_longitude), (row['lat'], row['lng'])).km, axis=1
    )
    print(data['Distance'])
    # Filter places within 5 kilometers
    filtered_data = data[data['Distance'] <= 5]

    # Sort by similarity, distance, and rating
    sorted_data = filtered_data.sort_values(by=['Similarity', 'Rating'], ascending=[False, False])

    # Filter top 3 recommendations
    recommendations = sorted_data.head(3).to_dict('records')

    # Convert recommendations to JSON format
    recommendations_json = json.dumps(recommendations)

    print(recommendations_json)
    # Render the page3.html template with the recommendations
    return render_template('page3.html',  recommendations_json = json.dumps(recommendations))


@app.template_filter('json_loads')
def json_loads_filter(value):
    return json.loads(value)

if __name__ == "__main__":
    app.run(debug=True)
