from flask import Flask,request,render_template
import joblib
import pandas as pd
import numpy as np
import flask

gbm_pickle = joblib.load('lgb.pkl')

app = Flask(__name__)
          
    

@app.route('/')
def hello_world():
    return 'Hello World!'
    
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.files['file']
        data = pd.read_csv(f.filename,sep="\t")
        pred = np.exp(gbm_pickle.predict(data)) - 200
        result = dict(enumerate(pred.flatten(), 1))
        return render_template("success.html", name = result)
    except:
        return render_template("error.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)