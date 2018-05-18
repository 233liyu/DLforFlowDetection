# import tensorflow as tf

# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

import pandas as pd
import numpy as np
import re
#
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A1', 'A1', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3', 'B6', 'B9', 'B10']})
#
df3 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'C': ['C8', 'C9', 'C10', 'C11']})
print(df3.applymap(lambda x : int(x, 16)))
#
# result = pd.merge(df1, df3, on='A')
#
# print(result)

# a = pd.DataFrame([1, 1, 1, 2], columns=['one'])
# print(a)
# stri = 12
# d = pd.DataFrame([stri], columns=['one'])
# print(d)

def to_full_str(x):
    x = str(x)
    x = ('{:0<%d}' % int(50)).format(x)
    b = []
    l = len(x)
    for n in range(l):
        if n % 2 == 0:
            b.append(x[n:n+2])
    return '_'.join(b)

a = '12oihcds09a8yft023grb08hb0dsw8hrf'
print(to_full_str(a))
