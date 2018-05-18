import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

import pandas as pd
import numpy as np
#
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A1', 'A1', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3', 'B6', 'B9', 'B10']})
#
# df3 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                     'C': ['C8', 'C9', 'C10', 'C11']})
#
# result = pd.merge(df1, df3, on='A')
#
# print(result)

# a = pd.DataFrame([1, 1, 1, 2], columns=['one'])
# print(a)
# stri = 12
# d = pd.DataFrame([stri], columns=['one'])
# print(d)

df = pd.DataFrame({'A': ['40', '00', '0a', '3e', '25', 'a6', 'fa', '5d', '29', '10'],
                   'B': ['41', '00', '0a', '3e', '25', 'a6', 'fa', '5d', '29', '10']})

dfx=pd.DataFrame()

li1=[]
for i in df:
    n = 0
    q='s'+n.__str__()
    li1.append(q)


print(li1)

for i in df:
    n = 0
    # print(df[i])
    li=[]
    for j in df[i]:
        # df[i][j]=int(j,16)
        li.append({q:int(j,16)})

    print(li)

    dfx.append(dict(zip()), ignore_index=True)

print(dfx)
# print(df.A['40'])
# df2 = df.apply(lambda x: int(x, 16), axis=1)
# print(df2)

# print(int('0a', 16))