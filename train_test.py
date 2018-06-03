import pandas as pd
import os
import re
import ReadFiles
import platform

if platform.system() == "Darwin":
    data_set_root = '/Users/lee/Desktop/大四/毕业设计/GP_data'
elif platform.system() == "Linux":
    data_set_root = '/home/lee/Desktop/GP/linux'
else:
    data_set_root = '$HOME'


sample_size = 3000


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


def gra_write_split_hex2int(total_size, tmp_file_name, final_file_name):
    skip_rows = 0
    print("start to process "
          "from, ", tmp_file_name,
          " to ", final_file_name,
          "total size: ", total_size)
    print("remove final file: ", final_file_name)
    try:
        os.remove(os.path.join(data_set_root, final_file_name))
    except IOError:
        print("final not existed")
    else:
        print("removed")

    while total_size > 0:
        read_size = 10000
        if total_size < read_size:
            read_size = total_size
        tr_set = pd.read_csv(os.path.join(data_set_root, tmp_file_name),
                             header=None,
                             skiprows=skip_rows,
                             nrows=read_size,
                             index_col=0)
        total_size -= read_size
        skip_rows += read_size
        print("read ", read_size, "rows, ", total_size, "rows left to read")
        tr_set = tr_set[1].str.split('_', expand=True)
        tr_set = tr_set.applymap(lambda x: int(x, 16))
        print("writing into ", final_file_name)
        with open(os.path.join(data_set_root, final_file_name), 'a+') as f:
            tr_set.to_csv(f, header=False)

    print("remove tmp file: ", tmp_file_name)
    os.remove(os.path.join(data_set_root, tmp_file_name))
    print("---------------process finished----------------")


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
    print('merge finished')

    # cutting data set
    label_count = train.app_protocol.value_counts(sort=True)
    print(label_count)

    # # drop unknown set
    print("process unknown protocol")
    unknown_set = train.loc[train.app_protocol == 'Unknown']
    train = train[train.app_protocol != 'Unknown']
    unknown_set = unknown_set.drop(columns=['file_id', 'app_protocol'])
    unknown_set.to_csv(os.path.join(data_set_root, 'tmp_unknown.csv'), sep=',', header=None)
    gra_write_split_hex2int(len(unknown_set), 'tmp_unknown.csv', 'handled_unknown.csv')
    print("unknown set dropped, and saved to handled_unknown.csv")

    label_count = label_count[label_count.index != 'Unknown']
    for index, value in label_count.iteritems():
        if value < 1000:
            # data set is too small
            train = train[train.app_protocol != index]
            print("protocol ", index, " removed, size:", value)
        elif value > sample_size:
            # too big
            temp = train.loc[train.app_protocol == index]
            train = train[train.app_protocol != index]
            temp = temp.sample(sample_size)
            print(temp)
            train = pd.concat([train, temp])

    # train = train[train.app_protocol != 'sslocal']
    train = train[train.app_protocol != 'DNS']
    train = train[train.app_protocol != 'QUIC']
    train = train[train.app_protocol != 'SSL']
    train = train[train.app_protocol != 'BitTorrent']
    # train = train[train.app_protocol != 'HTTP']
    train = train[train.app_protocol != 'HTTP_Download']

    labels = train.app_protocol
    train = train.drop(columns=['file_id', 'app_protocol'])

    print(train['packet'])

    # causing memory shortage
    # train = train['packet'].str.split('_', expand=True)

    print("labels write to file")
    labels.to_csv(os.path.join(data_set_root, 'handled_labels.csv'), sep=',')
    labels = None
    print("labels write to file finished")

    print("train tmp save")
    train.to_csv(os.path.join(data_set_root, 'tmp.csv'), sep=',', header=None)
    print("train tmp save finished")

    print("start to split and convert")
    gra_write_split_hex2int(len(train), 'tmp.csv', 'handled_train.csv')
    print('split finished')
