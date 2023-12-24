from flask import Flask,request,jsonify
import pickle
import numpy as np 
import pandas as pd

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('rf_model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/')
def home() : 
    return 'hello world'

@app.route('/predict',methods=['POST'])
def predict() : 
        prediction = -1
        if request.method == 'POST':
            pregs = int(request.form.get('pregs'))
            gluc = int(request.form.get('gluc'))
            bp = int(request.form.get('bp'))
            skin = int(request.form.get('skin'))
            insulin = float(request.form.get('insulin'))
            bmi = float(request.form.get('bmi'))
            func = float(request.form.get('func'))
            age = int(request.form.get('age'))
            
            input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
            prediction = model.predict(scaler.transform(input_features))
            return jsonify({'outcome':int(prediction)})
        
    
   
    
    

if __name__ == '__main__' : 
    app.run(debug=True)