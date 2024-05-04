# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from joblib import dump, load

# Loading DataSet
df = pd.read_csv('cyber.csv')
states = list(set(df['State'].to_list()))
states.sort()

for state in states:
    # Preparing Data For Single State
    one_state = df[df['State'] == state]
    X = one_state.drop(['Unique Code', 'State', 'Cases Reported'], axis=1)
    y = one_state['Cases Reported']

    # With Complete Data
    final_model = XGBRegressor()
    final_model.fit(X.values, y.values)
    predictions = final_model.predict(X.values)

    # Exporting and Importing.
    path = f'models/{state}_xgboost.joblib'
    dump(final_model, path)
    loaded_model = load(path)

    # Testing on New Input
    sample_input = [[2022, 90000000, 60000000, 57000000, 59000000]]
    result = loaded_model.predict(sample_input)
    print(result[0])
