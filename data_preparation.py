
#https://github.com/mohit1997/DeepZip
import sys
import numpy as np
import json
import argparse
from sklearn.preprocessing import OneHotEncoder

def process_data(file_name):
    with open(file_name) as fp:
        data = fp.read()

    print(len(data))
    vals = list(set(data))
    char2id_dict = {c: i for (i,c) in enumerate(vals)}
    id2char_dict = {i: c for (i,c) in enumerate(vals)}

    params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}

    print(char2id_dict)
    print(id2char_dict)

    out = [char2id_dict[c] for c in data]
    integer_encoded = np.array(out)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    print(integer_encoded[:10])
    print(data[:10])
    return integer_encoded, params

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)
    
def generate_single_output_data(series,batch_size,time_steps):
    print(series.shape)
    series = series.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit(series)

    series = series.reshape(-1)

    data = strided_app(series, time_steps+1, 1)
    l = int(len(data)/batch_size) * batch_size

    data = data[:l] 
    print(data.shape)
    X = data[:, :-1]
    Y = data[:, -1:]

    Y = onehot_encoder.transform(Y)
    return X,Y