# imports
import pickle
from flask import Flask, jsonify, request

# defining a flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    
    # loading the model using pickle
    with open(file="model/model.bin", mode="rb") as model_file:
        
        model = pickle.load(model_file)
    
    # loading the DictVectorizer using pickle    
    with open(file="model/dv.bin", mode="rb") as dv_file:
        
        dv = pickle.load(dv_file)
        
    client = request.get_json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    
    result = {"heart_failure_prediction_probability": y_pred}
    
    return jsonify(result)

if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=4242)
