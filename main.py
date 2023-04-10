import dill
import json
import os
import pandas as pd


def main():
    print('And that is how the program is started for the name of abyssal divinities')
    model_filename = 'data/models/cars_pipe_202304031836.pkl'
    path = 'data/test/'
    test_list = os.listdir(path)
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    result_dict = dict()
    for test_file in test_list:
        with open(path+test_file) as file:
            x = pd.DataFrame([json.load(file)])
            y = model.predict(x)
            result_dict[test_file] = y
            print(f'test: {test_file}; result: {y[0]}')
    print('The program is done, and there is no other instructions, so the calculations end and we are staying silent')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
