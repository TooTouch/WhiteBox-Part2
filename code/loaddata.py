import pandas as pd 
import numpy as np 
import os

from sklearn.preprocessing import StandardScaler

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


    
        
