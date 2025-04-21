import re
import math
import os
import pickle
import pandas as pd
from urllib.parse import urlparse, urlunparse
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phishing_ensemble_models.pkl'))
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['best_model']  # Extract model from dictionary
        print(f"Model class: {model.__class__.__name__}")
        print(f"Expected features: {model.n_features_in_}")
except Exception as e:
    print(f"Model loading error: {str(e)}")
    raise

def validate_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = f'http://{url}'
            parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Missing domain")
        return urlunparse(parsed._replace(fragment=''))
    except Exception as e:
        app.logger.error(f"URL validation failed: {str(e)}")
        return None

# Helper functions
def parse_url_components(url):
    parsed = urlparse(url)
    return {
        'full_url': url,
        'domain': parsed.netloc,
        'subdomains': parsed.netloc.split('.') if parsed.netloc else [],
        'path': parsed.path,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def calculate_entropy(text):
    if not text:
        return 0.0
    entropy = 0.0
    for char in set(text):
        p = text.count(char) / len(text)
        entropy -= p * math.log2(p)
    return entropy

def count_special_chars(text):
    special_chars = r"!\"#\$%&'\(\)\*\+,/:;<=>?@\[\\\]^`\{\|\}~"
    return len(re.findall(f'[{special_chars}]', text))

def has_repeated_digits(text):
    return int(bool(re.search(r'(\d)\1{1,}', text)))

# Feature extraction functions
def extract_features(url):
    components = parse_url_components(url)
    domain = components['domain']
    subdomains = components['subdomains']
    
    features = {
        # URL-based features
        'url_length': len(url),
        'number_of_dots_in_url': url.count('.'),
        'having_repeated_digits_in_url': has_repeated_digits(url),
        'number_of_digits_in_url': len(re.findall(r'\d', url)),
        'number_of_special_char_in_url': count_special_chars(url),
        'number_of_hyphens_in_url': url.count('-'),
        'number_of_underline_in_url': url.count('_'),
        'number_of_slash_in_url': url.count('/') + url.count('\\'),
        'number_of_questionmark_in_url': url.count('?'),
        'number_of_equal_in_url': url.count('='),
        'number_of_at_in_url': url.count('@'),
        'number_of_dollar_in_url': url.count('$'),
        'number_of_exclamation_in_url': url.count('!'),
        'number_of_hashtag_in_url': url.count('#'),
        'number_of_percent_in_url': url.count('%'),
        
        # Domain-based features
        'domain_length': len(domain),
        'number_of_dots_in_domain': domain.count('.'),
        'number_of_hyphens_in_domain': domain.count('-'),
        'having_special_characters_in_domain': int(count_special_chars(domain) > 0),
        'number_of_special_characters_in_domain': count_special_chars(domain),
        'having_digits_in_domain': int(bool(re.search(r'\d', domain))),
        'number_of_digits_in_domain': len(re.findall(r'\d', domain)),
        'having_repeated_digits_in_domain': has_repeated_digits(domain),
        
        # Subdomain features
        'number_of_subdomains': max(len(subdomains)-1, 0),
        'having_dot_in_subdomain': int(any('.' in sub for sub in subdomains)),
        'having_hyphen_in_subdomain': int(any('-' in sub for sub in subdomains)),
        'average_subdomain_length': sum(len(sub) for sub in subdomains)/len(subdomains) if subdomains else 0,
        'average_number_of_dots_in_subdomain': sum(sub.count('.') for sub in subdomains)/len(subdomains) if subdomains else 0,
        'average_number_of_hyphens_in_subdomain': sum(sub.count('-') for sub in subdomains)/len(subdomains) if subdomains else 0,
        'having_special_characters_in_subdomain': int(any(count_special_chars(sub) > 0 for sub in subdomains)),
        'number_of_special_characters_in_subdomain': sum(count_special_chars(sub) for sub in subdomains),
        'having_digits_in_subdomain': int(any(bool(re.search(r'\d', sub)) for sub in subdomains)),
        'number_of_digits_in_subdomain': sum(len(re.findall(r'\d', sub)) for sub in subdomains),
        'having_repeated_digits_in_subdomain': int(any(has_repeated_digits(sub) for sub in subdomains)),
        
        # Path and query features
        'having_path': int(bool(components['path'])),
        'path_length': len(components['path']),
        'having_query': int(bool(components['query'])),
        'having_fragment': int(bool(components['fragment'])),
        'having_anchor': int(bool(components['fragment'])),
        
        # Entropy features
        'entropy_of_url': calculate_entropy(url),
        'entropy_of_domain': calculate_entropy(domain)
    }
    
    return pd.DataFrame([features])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = None  # Initialize url variable
        
        if not data or 'url' not in data:
            app.logger.error("No URL provided in request")
            return jsonify({"error": "Missing URL parameter"}), 400
            
        raw_url = data.get('url', '').strip()
        if not raw_url:
            app.logger.error("Empty URL provided")
            return jsonify({"error": "Empty URL provided"}), 400

        # Validate and normalize URL
        url = validate_url(raw_url)
        if not url:
            app.logger.error(f"Invalid URL format: {raw_url}")
            return jsonify({"error": "Invalid URL format"}), 400

        # Feature extraction
        features = extract_features(url)
        
        # Check feature compatibility
        if features.shape[1] != model.n_features_in_:
            app.logger.error(f"Feature mismatch. Model expects {model.n_features_in_}, got {features.shape[1]}")
            return jsonify({"error": "Model compatibility error"}), 500

        # Prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][prediction]
        
        return jsonify({
            "prediction": "phishing" if prediction == 1 else "legitimate",
            "confidence": round(confidence * 100, 2)
        })
        
    except Exception as e:
        # Use url variable safely
        error_url = url or raw_url or "unknown"
        app.logger.error(f"Error processing URL '{error_url}': {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
