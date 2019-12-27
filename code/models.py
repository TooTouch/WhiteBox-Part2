from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import tensorflow as tf

import pickle
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Classifier:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        if model_type == 'LR': # Logistic Regression
            # modeling
            self.model = LogisticRegression(**kwargs)
        elif model_type == 'DT': # Decision Tree
            # modeling
            self.model = DecisionTreeClassifier(**kwargs)
        elif model_type == 'RF': # Random Forest
            # modeling    
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == 'XGB': # XGboost
            # default
            kwargs['objective'] = "binary:logistic"
            # modeling
            self.model = XGBClassifier(**kwargs)
        elif model_type == 'DNN': # Deep Neural Network
            # default params
            nb_features = kwargs['nb_features'] if 'nb_features' in kwargs.keys() else NameError("name 'nb_features' is not defined")
            nb_class= kwargs['nb_class'] if 'nb_class' in kwargs.keys() else NameError("name 'nb_class' is not defined")
            nb_layers = kwargs['nb_layers'] if 'nb_layers' in kwargs.keys() else NameError("name 'nb_layers' is not defiend")
            loss = 'categorical_crossentropy' if nb_class > 2 else 'binary_crossentropy'
            act_func = 'softmax' if nb_class > 2 else 'sigmoid'

            # modeling
            input_ = tf.keras.layers.Input(shape=(nb_features,))
            x = input_
            for i in range(len(nb_layers)):
                x = tf.keras.layers.Dense(nb_layers[i], activation='relu')(x)
            output = tf.keras.layers.Dense(nb_class, activation=act_func)(x)
            self.model = tf.keras.models.Model(input_, output)

            # complie
            self.model.compile(optimizer=kwargs['optimizer'], 
                               loss=loss, 
                               metrics=['acc'])

    def train(self, X, y, savedir=None, **kwargs):
        # set evaluation dataset when model selected as XGB
        if self.model_type == 'XGB':
            kwargs['eval_set'] = [(X,y)]

        # model training
        self.model.fit(X, y, **kwargs)

        # save model
        if savedir!=None:
            # check save directory
            if not os.path.isdir('../saved_models'):
                os.mkdir('../saved_models')    
            # model save to pickle except DNN
            if self.model_type=='DNN':
                self.model.save(savedir)
            else:
                pickle.dump(self.model, open(savedir,"wb"))

        
class Regressor:
    def __init__(self, model_type, **kwargs):
        if model_type == 'LR': # Logistic Regression
            # modeling
            self.model = LinearRegression(**kwargs)
        elif model_type == 'DT': # Decision Tree
            # modeling
            self.model = DecisionTreeRegressor(**kwargs)
        elif model_type == 'RF': # Random Forest
            # modeling
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'XGB': # XGboost
            # default
            kwargs['objective'] = "reg:linear"
            # modeling
            self.model = XGBRegressor(**kwargs)
        elif model_type == 'DNN' : # Deep Neural Network
            # default params
            nb_features = kwargs['nb_features'] if 'nb_features' in kwargs.keys() else NameError("name 'nb_features' is not defined")
            nb_layers = kwargs['nb_layers'] if 'nb_layers' in kwargs.keys() else NameError("name 'nb_layers' is not defiend")

            # modeling
            input_ = tf.keras.layers.Input(input_shape=(nb_features,))
            x = input_
            for i in range(len(nb_layers)):
                x = tf.keras.layers.Dense(nb_layers[i], activation='relu')(x)
            output = tf.keras.layers.Dense(1)(x)
            self.model = tf.keras.models.Model(input_, output)

            # complie
            self.model.compile(optimizer=kwargs['optimizer'], 
                               loss='mse')
                               
    def train(self, X, y, savedir=None, **kwargs):
        self.model.fit(X, y, **kwargs)

        if savedir!=None:
            # check save directory
            if not os.path.isdir('../saved_models'):
                os.mkdir('../saved_models')    
            # save model
            if self.model_type=='DNN':
                self.model.save(savedir)
            else:
                pickle.dump(self.model, open(savedir,"wb"))

        