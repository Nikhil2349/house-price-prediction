from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder='.')
data=pd.read_csv('banglore house cleaned data.csv')
pipe=pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('sqft')

    print(location,bhk,bath,sqft)
    input_data = pd.DataFrame([[location, float(bhk), float(bath), float(sqft)]], columns=['location', 'bhk', 'bath', 'total_sqft'])
    print("Input Data:", input_data)
    prediction=pipe.predict(input_data)[0] * 100000
    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True, port=5881)