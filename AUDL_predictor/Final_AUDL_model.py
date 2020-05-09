#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler


fts = pd.read_csv('teamStats.csv')
fts.set_index('Team',inplace=True)

#all_stat = pd.read_csv('FULL_DATAFRAME')

all_stat.drop('Unnamed: 0',axis=1,inplace=True)

mdl = load_model("AUDL_regression.h5")

#just for rfe
def predict_game_rfe(home, away):
    """
    Predicts the outcome between any two teams using the rfe NN built in previous notebook
    """
    mdl = load_model('AUDL_RFE_82.h5')

    home_cols = ['W', 'L', 'PPG', 'PAPG', 'PM', 'BPG', 'CMP', 'CPG','TPG']
    h_stat = fts.loc[home].loc[home_cols]
    away_cols = ['G', 'W', 'L', 'PPG', 'PAPG', 'OEFF', 'DEFF','PMPG', 'BLK', 'BPG', 'CMP', 'CPG', 'CMPR', 'TO','TPG']
    a_stat = fts.loc[away].loc[away_cols]

    f_inp = np.array(h_stat.append(a_stat)).reshape(-1,24)

    pred = mdl.predict_proba(f_inp).reshape(-1)[0]
    outcome_str = str(round(pred*100,4)) + "% chance "+home+' wins'

    if pred < .5:
        outcome_str = str(100-round(pred*100,3)) + "% chance "+away+' wins'

    return outcome_str




#for model that takes in all 38 features
def predict_game_fullfeats(home,away):
    """
    Predicts the probability of any two teams winning
    """
    mdl = load_model('ANN_classifier_82.h5')

    #fetch stat line for each team
    home_stat = np.array(fts.loc[home])
    away_stat = np.array(fts.loc[away])
    game_stat = np.append(home_stat,away_stat).reshape(1,38)

    pred = mdl.predict_proba(game_stat).reshape(-1)[0]
    outcome_str = str(round(pred*100,4)) + "% chance "+home+' wins'

    if pred < .5:
        outcome_str = str(100-round(pred*100,3)) + "% chance "+away+' wins'

    return outcome_str


def predict_game_score(home,away):
    #open linear regression model
    with open('AUDL_linear_model','rb') as f:
        mdl = pickle.load(f)
    #fetch team stats
    home_stat = np.array(fts.loc[home])
    away_stat = np.array(fts.loc[away])
    game_stat = np.append(home_stat,away_stat).reshape(1,38)

    pred = int(mdl.predict(game_stat)[0])

    outcome_str = home+' wins by a margin of '+str(pred)

    if pred < 0:
        outcome_str = away+' wins by a margin of '+str(abs(pred))

    return outcome_str
