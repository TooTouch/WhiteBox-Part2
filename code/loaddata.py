import pandas as pd 
import numpy as np 
import os

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

class TitanicData:
    def __init__(self, file_path):
        self.data = pd.read_csv(os.path.join(file_path,'train.csv'))
        self.testset = pd.read_csv(os.path.join(file_path,'test.csv'))

        self.scaler = StandardScaler()
        self.num_features = ['Pclass','Age','SibSp','Parch','Fare']

    def transform(self, **kwargs):
        # args
        scaling = False if 'scaling' not in kwargs.keys() else kwargs['scaling']

        # pre-processing
        train = self.processing(self.data, **kwargs)
        x_train = train.drop('Survived', axis=1)
        y_train = train.Survived
        # scaling
        if scaling:
            x_train[self.num_features] = self.scaler.fit_transform(x_train[self.num_features])

        # test set
        if isinstance(self.testset, pd.DataFrame):
            x_test = self.processing(self.testset, **kwargs)
            # scaling
            if scaling:
                x_test[self.num_features] = self.scaler.transform(x_test[self.num_features])
    
            return (x_train.values, y_train.values), x_test.values

        return x_train.values, y_train.values
        
    def processing(self, raw_data, **kwargs):
        data = raw_data.copy()
        # args
        dummy_dropfirst = True if 'dummy_dropfirst' not in kwargs.keys() else kwargs['dummy_dropfirst']

        # Sex은 0,1 로 변환
        sex_dict = {
            'male': 0,
            'female': 1
        }
        data['Sex'] = data.Sex.map(sex_dict)

        # Name은 Title을 추출하여 ['Mr','Mrs','Miss','Master','Other'] 로 분류
        # Title에 대한 정보 : https://en.wikipedia.org/wiki/Honorific
        data.Name = data.Name.str.split('.', expand=True).iloc[:,0].str.split(',', expand=True).iloc[:,1].str.strip()
        major = data.Name.value_counts().iloc[:4]
        data.Name = data.Name.apply(lambda x : 'Other' if x not in major.index else x)

        # Age는 각 타이틀별 중앙값으로 대체
        age_median = dict(data.groupby('Name').Age.median())
        for k, v in age_median.items():
            data.loc[data.Age.isnull() & (data.Name==k), 'Age'] = v
        # 왜인지 모르겠지만 age에 소수점이 있음
        data['Age'] = data.Age.astype(int)

        # Embarked는 최빈값으로 대체    
        data.loc[data.Embarked.isnull(), 'Embarked'] = data.Embarked.mode().values

        # Fare는 큰 이상치가 있기때문에 log1p 변환
        data.loc[data.Fare.isnull(), 'Fare'] = data.Fare.median()
        data['Fare'] = np.log1p(data.Fare)

        # Tickek과 Cabin은 사용안함
        data = data.drop(['Ticket','Cabin'], axis=1)

        # PassengerId 제외
        data = data.drop('PassengerId', axis=1)

        # dummy transform
        data = pd.get_dummies(data, drop_first=dummy_dropfirst)

        return data


    
class HousePriceData:
    def __init__(self, file_path):
        self.data = pd.read_csv(os.path.join(file_path,'train.csv'))
        self.testset = pd.read_csv(os.path.join(file_path,'test.csv'))

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer()
        self.encoder = OneHotEncoder()
        self.num_features = None
        self.missing_features = None
        self.skew_features = None
        self.remove_features = []
        
    def transform(self, **kwargs):
        # args
        scaling = False if 'scaling' not in kwargs.keys() else kwargs['scaling']

        # pre-processing
        train = self.processing(self.data, **kwargs)
        x_train = train.drop('SalePrice', axis=1)
        y_train = train.SalePrice
        
        # test set
        x_test = self.processing(self.testset, training=False, **kwargs)

        # dummy transform
        data = pd.concat([x_train, x_test],axis=0)
        data = pd.get_dummies(data, drop_first=False)

        # split train and test
        x_train = data.iloc[:x_train.shape[0]]
        x_test = data.iloc[x_train.shape[0]:]

        # imputation
        x_train.iloc[:,:] = self.imputer.fit_transform(x_train)
        x_test.iloc[:,:] = self.imputer.transform(x_test)

        # scaling
        if scaling:
            x_train[self.num_features] = self.scaler.fit_transform(x_train[self.num_features])
            x_test[self.num_features] = self.scaler.transform(x_test[self.num_features])

        return (x_train.values, y_train.values), x_test.values
        
    def processing(self, raw_data, **kwargs):
        training = True if 'training' not in kwargs.keys() else False

        data = raw_data.copy()

        # Remove ID columns
        data = data.drop('Id',axis=1)
        
        if training:
            # Remove features
            # filtering features over 10% missing values
            missing_lst = data.isnull().mean().reset_index(name='pct')
            missing_features_over10 = missing_lst[missing_lst['pct'] >= 0.10]['index'].tolist()
            self.remove_features.extend(missing_features_over10)
            # filtering features over 10 unique values
            unique_lst = data.describe(include='all').loc['unique'].reset_index(name='cnt')
            unique_features = unique_lst[unique_lst.cnt >=10]['index'].tolist()
            self.remove_features.extend(unique_features)
            
            # Log 1+ Transform features over 0.75 skewness
            num_features = data.dtypes[data.dtypes!='object'].index
            skew_lst = data[num_features].apply(lambda x: skew(x.dropna())).reset_index(name='skew_value')
            self.skew_features = skew_lst[skew_lst.skew_value > 0.75]['index'].tolist()
            self.num_features = num_features.tolist()

            # remove target from skew features
            if 'SalePrice' in self.skew_features:
                self.skew_features.remove('SalePrice')
                self.num_features.remove('SalePrice')

            # remove deleted features from num features
            del_features = set(self.remove_features) & set(self.num_features)
            for f in del_features:
                self.num_features.remove(f)
            # remove deleted features from skew features
            del_features = set(self.remove_features) & set(self.skew_features)
            for f in del_features:
                self.skew_features.remove(f)
                
        # remove
        data = data.drop(self.remove_features, axis=1)

        # log transform
        data[self.skew_features] = np.log1p(data[self.skew_features])

        return data

