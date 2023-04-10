from datetime import datetime
import dill
import json
import logging
import pandas as pd
import os

path = os.environ.get('PROJECT_PATH', '.')


# Получаю последнюю по времени создания модель и загружаю ее
def getting_model():
    models = os.listdir(f'{path}/data/models/')
    print(models)
    models = [f'{path}/data/models/{model}' for model in models]
    print(models)
    model = max(models, key=os.path.getctime)
    print(model)

    logging.info(f'File is used {model}')

    with open(model, 'rb') as file:
        model = dill.load(file)

    logging.info(f'Model is used: {type(model.named_steps["classifier"]).__name__}')

    return model


# Получаю все json из data/test/ и записываю их в датафрейм
def getting_jsons():
    df = pd.DataFrame()

    for filename in os.listdir(f'{path}/data/test/'):
        with open(f'{path}/data/test/{filename}', 'r') as file:
            data = json.loads(file.read())
            data_df = pd.DataFrame([data])

        df = pd.concat([df, data_df], ignore_index=True)

    return df


# Делаю предсказания для всех объектов и сохраняю их в csv-формате
def predict():
    model = getting_model()
    df = getting_jsons()

    df['pred'] = model.predict(df)
    df = df.rename(columns={'id': 'car_id'}).set_index('car_id')[['pred']]

    filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df.to_csv(filename)

    logging.info(f'Prediction is saved as {filename}')


if __name__ == '__main__':
    predict()