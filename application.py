from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
region_abbes = "sidibelabbes"
region_bejaia = "bejaia"
# Models and assets path
modle_path = "models"
# model_lists = ['Algerian_fires_linear.pkl',\
#'Algerian_fires_ridgecv.pkl','Algerian_fires_scaler.pkl']
model_lists = ["Algerian_fires_linear.pkl", "Algerian_fires_scaler.pkl"]



def load_pickles(model_name,scaler_name):
    try:
        print(f"debug:{modle_path},{model_name}")
        model = pickle.load(open(f'{modle_path}/{model_name}', "rb"))
        scaler = pickle.load(open(f'{modle_path}/{scaler_name}', "rb"))
        return (model, scaler)
    except Exception as e:
        print(e)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    print("starting prediction")
    regressor, scaler = load_pickles( "Algerian_fires_linear.pkl","Algerian_fires_scaler.pkl")
    if request.method == "GET":
        return render_template("prediction.html")
    if request.method == "POST":
        isi = float(request.form.get('ISI'))                     
        rh = float(request.form.get('RH'))                     
        dmc = float(request.form.get('DMC'))                      
        ffmc = float(request.form.get('FFMC'))                      
        temperature = float(request.form.get('Temperature'))                      
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))  
        
        classes_fire = request.form.get('fire')
        if classes_fire == None:
            classes_not_fire = 1
            classes_fire = 0
        else:
            classes_fire = 1
            classes_not_fire = 0



        region = request.form.get('region').replace(' ','').lower()
        if region == region_abbes:
            sidi_bel_abbes = 1
            bejaia = 0
        if region == region_bejaia:
            bejaia = 1 
            sidi_bel_abbes = 0
        
        # Scale the incoming features
    scaled_feature = scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,bejaia,sidi_bel_abbes,classes_fire,classes_not_fire]])
    prediction = regressor.predict(scaled_feature)
    return render_template("prediction.html",result=prediction[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, use_debugger=False, use_reloader=False)
