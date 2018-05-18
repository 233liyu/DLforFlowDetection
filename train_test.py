import pandas as pd
import os
import re
import ReadFiles

data_set_root = '/Users/lee/Desktop/大四/毕业设计/GP_data'


def read_label(result_file_list):
    df = pd.DataFrame({}, columns=['file_id', 'ms_protocol', 'app_protocol', 'process'])
    for file_name in result_file_list:
        lb_csv = pd.read_csv(file_name,
                             names=['file_id', 'ms_protocol', 'app_protocol', 'process'],
                             header=None)
        # concat all the label files
        df = pd.concat([df, lb_csv])
        # drop 2 columns
    return df.drop(columns=['ms_protocol', 'process'])


def read_data(data_file_list):
    df = pd.DataFrame({}, columns=['file_id', 'packet'])
    for file_name in data_file_list:
        data_csv = pd.read_csv(file_name,
                               names=['file_id', 'packet'],
                               header=None)
        # concat all the label files

        cont = data_csv.packet
        for fa in cont:
            fa = str(fa)
            if len(fa) < (ReadFiles.single_packet_length / 2):
                fa = '{:0<1568}'.format(fa)
        data_csv.packet = cont
        df = pd.concat([df, data_csv])
    return df


label_list = []
raw_list = []
for root, dirs, files in os.walk(data_set_root):
    if root == data_set_root:
        for file in files:
            if re.match('ndpi_result.*.csv', file):
                # is the label file
                label_list.append(os.path.join(data_set_root, file))
            elif re.match('payload_info.*.csv', file):
                # is the raw data file
                raw_list.append(os.path.join(data_set_root, file))
        print("label file found: ", len(label_list))
        print("data  file found: ", len(raw_list))

print("start to read labels...")
train_label = read_label(label_list)
print("label reading finished, labels:", len(train_label))
print("start to read data...")
train_feature = read_data(raw_list)
print("data reading finished, data:", len(train_feature))

train = pd.merge(train_label, train_feature, on='file_id')
labels = train.app_protocol
train = train.drop(columns=['file_id', 'app_protocol'])

print(train['packet'][0])
train = train['packet'].str.findall(r'.{2}')
train.fillna('00')

print(train)
print('-------------------------------')

# name_list = []
# for i in range(748):
#     name_list.append('p%d' % i)
# train.rename(name_list)

# print(train.iloc[:, 0])
print(train.applymap(lambda x: int(x, 16)))

print('-------------------------------')


train_mod = pd.DataFrame()



for xx in train:
    li = []
    # print(xx)
    for yy in xx:
        # print(int(yy,16))
        li.append( int(yy, 16))
        # yy=1
    xx=li

print(train)
