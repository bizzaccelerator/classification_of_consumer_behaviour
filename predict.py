import pickle
from flask import Flask
from flask import request, jsonify


# Importing the model using pickle
model_file = 'random_forest_model_estimators=20_max_features=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv,model =pickle.load(f_in)


# Creating the Flask application
app = Flask('classification')

# Defining the route for the application
@app.route('/predict', methods=['POST'])

# The service I want to serve
def predict():
    customer = request.get_json()

    # A note for the user
    if not customer:
        return jsonify({'error': 'No input data provided'}), 400
    
    # The core logic
    try: 
        X = dv.transform([customer])
        y_pred = model.predict(X)

        result = {
            'The predicted class for this costumer is': int(y_pred[0]) + 1 
            # One is added to adjust for the original classes.
        }
    except Exception as e: # Error handling message
        return jsonify({'error': str(e)}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)