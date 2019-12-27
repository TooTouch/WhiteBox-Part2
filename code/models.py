from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import tensorflow as tf

class Classifier:
    def __init__(self, type, random_state=223, **kwargs):
        if type == 'LR': # Logistic Regression
            self.model = LogisticRegression(random_state=random_state)
        elif type == 'DT': # Decision Tree
            self.model = DecisionTreeClassifier(random_state=random_state)
        elif type == 'RF' # Random Forest
            self.model = RandomForestClassifier(random_state=random_state)
        elif type == 'XGB': # XGboost
            self.model = XGBClassifier(random_state=random_state)
        elif type == 'DNN' : # Deep Neural Network
            nb_features = kwargs['nb_features'] if 'nb_features' in kwargs.keys() else raise NameError("name 'nb_features' is not defined"):
            nb_features = kwargs['nb_class'] if 'nb_class' in kwargs.keys() else raise NameError("name 'nb_class' is not defined"):

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, input_shape=(nb_features,), activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(nb_class, activation='softmax')
            ])

            self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y):
        self.model.fit(X, y, **kwargs)

        # TODO: 집가서 모델 테스트, DNN의 경우 Model save 및 재현성 잘 확인

        
class Regressor:
    def __init__(self, type, **kwargs):
        if type == 'LR': # Logistic Regression
            self.model = LinearRegression(random_state=random_state)
        elif type == 'DT': # Decision Tree
            self.model = DecisionTreeRegressor(random_state=random_state)
        elif type == 'RF' # Random Forest
            self.model = RandomForestRegressor(random_state=random_state)
        elif type == 'XGB': # XGboost
            self.model = XGBRegressor(random_state=random_state)
        elif type == 'DNN' : # Deep Neural Network
            nb_features = kwargs['nb_features'] if 'nb_features' in kwargs.keys() else raise NameError("name 'nb_features' is not defined"):

            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, input_shape=(nb_features,), activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X, y):
        self.model.fit(X, y, **kwargs)

        # TODO: 집가서 모델 테스트
        