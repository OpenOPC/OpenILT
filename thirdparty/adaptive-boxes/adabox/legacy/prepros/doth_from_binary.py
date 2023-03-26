import sys
import numpy as np

if len(sys.argv) < 3:
    print('ERROR args. Needed \n[1]in_path(.binary) \n[2]out_path(.h)')
    sys.exit()

in_path = str(sys.argv[1])      # .binary
out_path = str(sys.argv[2])     # .h


# in_path = '/Users/Juan/django_projects/adaptive-boxes/data_prepros/complex.binary'
# out_path = '/Users/Juan/django_projects/adaptive-boxes/data_cpp/complex.h'

print("Working...")
data_matrix = np.loadtxt(in_path, delimiter=",")

data_m = data_matrix.shape[0]
data_n = data_matrix.shape[1]

np.array_str(data_matrix.flatten())

text_file = open(out_path, "w")
text_file.write('long m = %d; \nlong n = %d; \n\n' % (data_m, data_n))
text_file.write('int data[%ld] = { \n' % (data_m*data_n))


for i in range(data_m):
    for j in range(data_n):
        text_file.write('%d, ' % data_matrix[i][j])
    text_file.write('\n')


text_file.write('};\n\n')
text_file.close()

print("Work Finished!!")
