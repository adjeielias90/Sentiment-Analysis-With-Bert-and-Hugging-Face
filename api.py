from flask import Flask, request, redirect, url_for, flash, jsonify
from sentiment_analyzer import get_prediction_as_json
from sentiment_analyzer_dataset import get_prediction_as_json_with_id
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

#Start API with api.py


app = Flask(__name__)
CORS(app)

'''
  Our api test path.
  Returns just a simple string to indicate our API is running
  and ready to serve requests.

  example request:
  http://0.0.0.0:5000/api
'''
@app.route('/api/', methods=['GET'])
def api_root():
  prediction = 'message: API root, nothing here'
  return jsonify(prediction)

@app.route('/api/review/<id>', methods=['GET'])
def get_top_movies_by_genre(id):
  id = int(id)
  prediction = get_prediction_as_json_with_id(id)
  return prediction

@app.route('/api/review/prediction', methods=['POST'])
def get_prediction_with_review():
  data = request.get_json()
  review = data.get('review', '')
  prediction = get_prediction_as_json(review)

  return prediction


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')