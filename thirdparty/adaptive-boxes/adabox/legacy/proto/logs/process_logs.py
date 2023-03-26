
import numpy as np

path = '/Users/Juan/django_projects/adaptive-boxes/proto/logs/data'
in_file = 'log_boston12.csv_1'

file = open(path + '/' + in_file, 'r')
data_str = file.read()

data_split = data_str.split('\n')

values = []
for d in data_split:
    index = d.find("-->Elapsed time:")
    if index != -1:
        val = float(d[index + 16: -1])
        # print(val)
        values.append(val)

a = np.array(values)

print(data_split)
print(a.mean())
