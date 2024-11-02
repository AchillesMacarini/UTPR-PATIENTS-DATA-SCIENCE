import os
import pandas as pd

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '..', 'data-set', 'patients-data.csv')

data = pd.read_csv(file_path)
classes = data['PatientID']
features = data.drop(columns=['PatientID'])

print(data.head)