import pandas as pd
import os
path = 'M'
def combine(filename):
    a = pd.read_csv(f'./test_results/{path}/{filename}.csv').to_dict()
    b = pd.read_csv(f'./test_results/{path}/{filename}_more.csv').to_dict()
    a.update(b)
    del a['Unnamed: 0']
    pd.DataFrame(a).to_csv(f'./test_results_combined/{path}/{filename}.csv')


files = []
for filename in os.listdir(f'./test_results/{path}'):
    if filename[-9:] != '_more.csv':
        files.append(filename[:-4])

for filename in files:
    combine(filename)