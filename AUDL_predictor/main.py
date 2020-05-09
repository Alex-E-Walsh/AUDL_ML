import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('AUDL_linear_model.pkl','rb'))
fts = pd.read_csv('AUDL_team_stats.csv')
fts.set_index('team',inplace=True)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    #get user inputs
    int_features = [str(x) for x in request.form.values()]
    home = int_features[1]+int_features[0]
    away = int_features[3]+int_features[2]

    #fetch respective data from csv
    home_stat = np.array(fts.loc[home])
    away_stat = np.array(fts.loc[away])
    game_stat = np.append(home_stat,away_stat).reshape(1,38)

    pred = int(model.predict(game_stat)[0])


    return render_template('index.html',prediction_text=(pred))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    return jsonify(data)

#     fts = pd.read_csv('teamStats.csv')
#
#     home_stat = np.array(fts.loc[home])
#     away_stat = np.array(fts.loc[away])
#     game_stat = np.append(home_stat,away_stat).reshape(1,38)
#
#
# def predict_game_score(home,away):
#     #open linear regression model
#     with open('AUDL_linear_model','rb') as f:
#         mdl = pickle.load(f)
#     #fetch team stats
#     home_stat = np.array(fts.loc[home])
#     away_stat = np.array(fts.loc[away])
#     game_stat = np.append(home_stat,away_stat).reshape(1,38)
#
#     pred = int(mdl.predict(game_stat)[0])
#
#     outcome_str = home+' wins by a margin of '+str(pred)
#
#     if pred < 0:
#         outcome_str = away+' wins by a margin of '+str(abs(pred))
#
#     return outcome_str

if __name__ =="__main__":
    app.run(debug=True)
