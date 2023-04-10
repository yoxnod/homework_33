# <YOUR_IMPORTS>
import dill
import pandas as pd
import json
import os
import logging
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def pick_model():
#    model_path = f'{path}/data/models/'  # path to folder containing models (pkl)
#    model_list = os.listdir(model_path)  # list of models in the folder
#    file_model = model_path + model_list[-1]  # pick the last version
#    with open(file_model, 'rb') as file:
#        model = dill.load(file)  # load model
#    logging.info(f'File is used {file_model}')
#    return model
    models = os.listdir(f'{path}/data/models/')
    models = [f'{path}/data/models/{model}' for model in models]
    model = max(models, key=os.path.getctime)

    logging.info(f'File is used {model}')

    with open(model, 'rb') as file:
        model = dill.load(file)

    logging.info(f'Model is used: {type(model.named_steps["classifier"]).__name__}')

    return model


def grab_tests():
    path_test = f'{path}/data/test/'  # path to folder containing test json-files
    test_list = os.listdir(path_test)  # list of test json-files
    df_test = pd.DataFrame()
    for test_file in test_list:
        with open(path_test+test_file) as file:
            x = pd.DataFrame([json.load(file)]) # converting json to pandas dataframe
        df_test = pd.concat([df_test, x], ignore_index=True)
    return df_test


def predict():
    # <YOUR_CODE>
    model = pick_model()
    df = grab_tests()
    df['pred'] = model.predict(df)
    df_save = df[['id', 'pred']]
    #result_dict = dict()    # the variable to save results
    #for test_file in test_list:
    #    with open(path_test+test_file) as file:
    #        x = pd.DataFrame([json.load(file)]) # converting json to pandas dataframe
    #    result_dict[test_file] = model.predict(x)    # application of the model to predict car price category and saving result
    #    print(f'test: {test_file}; result: {y[0]}') # print to check
    #df = pd.DataFrame.from_dict(result_dict)    # converting dict to pandas dataframe
    #pred_filename =
    df_save.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')  # saving as csv
    logging.info(f'Prediction is saved.')

if __name__ == '__main__':
    predict()
