import pandas as pd
import json


def find_best_model():
    model_output_df = pd.read_csv("./log/model_output.log", delimiter='|',
                                  names=['level', 'datetime', 'experiment_id', 'trial_id', 'params', 'performance'])

    best_model = model_output_df.sort_values('performance', ascending=False).iloc[0][1:]

    with open(f'./config/best_model_metadata.json', 'w') as f:
        json.dump(dict(best_model), f)

    return best_model
