import pandas as pd
import os
import re
import ReadFiles
import platform

if platform.system() == "Darwin":
    data_set_root = '/Users/lee/Desktop/大四/毕业设计/GP_data'
elif platform.system() == "Linux":
    data_set_root = '/home/lee/Desktop/GP'
else:
    data_set_root = '$HOME'


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


def to_full_str(x):
    x = str(x)
    x = ('{:0<%d}' % int(ReadFiles.single_packet_length)).format(x)
    b = []
    l = len(x)
    for n in range(l):
        if n % 2 == 0:
            b.append(x[n:n + 2])
    return '_'.join(b)


def read_data(data_file_list):
    df = pd.DataFrame({}, columns=['file_id', 'packet'])
    for file_name in data_file_list:
        data_csv = pd.read_csv(file_name,
                               names=['file_id', 'packet'],
                               header=None)
        # concat all the data files
        cont = data_csv.packet.apply(to_full_str)
        #
        data_csv.packet = cont

        df = pd.concat([df, data_csv])
    return df


if __name__ == '__main__':
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

    # print(train['packet'])
    train = train['packet'].str.split('_', expand=True)

    train = train.applymap(lambda x: int(x, 16))
    print(train)

    print("writing to csv file")
    train.to_csv(os.path.join(data_set_root, 'handled_train.csv'), sep=',')
    labels.to_csv(os.path.join(data_set_root, 'handled_labels.csv'), sep=',')
