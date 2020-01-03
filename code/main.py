from loaddata import TitanicData
from models import Classifier, Regressor
import json
import os 
import argparse


def model_train(file_path, dataset, model_type, savedir, **kwargs):
    # load data
    # TODO: 아래 나머지 채울것
    if dataset=='titanic':
        titanic = TitanicData(file_path)
        (x_train, y_train), _ = titanic.transform(scaling=kwargs.pop('scaling'))
    elif dataset=='house_price':
        pass
    elif dataset=='bike_sharing':
        pass
    elif dataset=='cervical_canver':
        pass
    elif dataset=='youtube_spam':
        pass
    print('Complete Data Pre-processing')

    # add argument
    if model_type=='DNN':
        kwargs['params']['nb_features'] = x_train.shape[1]

    # model training
    clf = Classifier(model_type=model_type, **kwargs.pop('params'))
    clf.train(x_train, y_train, savedir, **kwargs)
    print('Complete Training Model')
    print('Complete Saving Model')


    
if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LR','DT','RF','XGB','DNN'], 
                        help='LR: Logistic Regression or Linear Regression / DT: Decision Tree / RF: Random Forest / XGB: XGBoost / DNN: Deep Neural Network')
    parser.add_argument('--dataset', type=str, choices=['titanic','house_price','bike_sharing','cervical_cancer','youtube_spam'], 
                        help='dataset category')
    parser.add_argument('--file_path',type=str, default='../dataset', help='dataset directory')
    parser.add_argument('--save_path',type=str, default='../saved_models',help='save model directory')
    parser.add_argument('--params_path',type=str, default='../params', help='model parameters directory')
    args = parser.parse_args()

    # file path
    file_path = os.path.join(args.file_path, args.dataset)
    # save path and name
    save_path = os.path.join(args.save_path, f'{args.dataset}_{args.model}')
    # load parameters 
    params_path = os.path.join(args.params_path, f'{args.model}_params.json')
    params = json.load(open(params_path, 'r'))
    

    # model training by dataset
    model_train(file_path, args.dataset, args.model, save_path, **params)
    
        