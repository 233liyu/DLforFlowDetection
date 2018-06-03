import os
import csv
import platform
import re
from collections import Counter

# data set dir
if platform.system() == "Darwin":
    root_dir = '/Users/lee/Desktop/大四/毕业设计/GP_data'
elif platform.system() == "Linux":
    root_dir = '/home/lee/Desktop/GP/linux'
ndpi_label_filename = 'ndpi_result.txt'
single_packet_length = 1568


# return sub dirs in the root dic
def sub_dir_list():
    for root, dirs, files in os.walk(root_dir):
        if root == root_dir:
            return dirs


# read from the raw data file and return a joint payload
# @return: [return payload, payload_length, process, success]
def read_raw_data(file_name):
    payload = ""
    process = "NULL"
    p_count = 0
    try:
        with open(file_name) as f:
            content = f.readlines()
    except IOError:
        print("file= ", file_name, "is not found")
        return '', process, p_count, False
    else:
        content = [x.strip() for x in content]
        read_type = 0
        for line in content:
            if read_type == 1:
                # joint the payload as one str
                payload += ',' + line.strip()
                read_type = 0
                continue
            if read_type == 2:
                # save the process info
                process = line[:line.find(' ')]
                read_type = 0
                continue

            if not line.find('src:'):
                read_type = 1
                p_count += 1
                # count the total length of the flow
                # payload_length += int(line[line.find("payload length:") + len("payload length:"):])
                continue
            if not line.find('COMMAND'):
                read_type = 2
                continue
        return payload, process, p_count, True


# read from the ndpi result data
# @return: list of tuples: flow name, Ms_protocol, app_protocol
def read_ndpi_result(ndpi_file_name):
    with open(ndpi_file_name) as f:
        content = f.readlines()
    # get rid of the \n
    content = [x.strip() for x in content]
    container = []
    # split into tuple
    for item in content:
        container.append(item.split('\t'))
    return container


def write_payload_to_csv(date_info, flow_name, payload, payload_count):
    csv_file_name = os.path.join(root_dir, 'payload_info.csv')
    pl = [tok for tok in re.split(',', payload) if len(tok) > 0]
    # print(pl)
    data = []
    # # cut the packet into single_packet_length
    # for pp in pl:
    #     # get the distribution to the payload length
    #     lene = len(pp)
    #     lene = int(lene / 10)
    #     payload_count[lene] += 1
    #
    #     if len(pp) >= single_packet_length:
    #         info_tuple = (str(date_info + ':' + flow_name), pp[:single_packet_length])
    #     else:
    #         info_tuple = (str(date_info + ':' + flow_name), pp)
    #     data.append(info_tuple)

    for i in range(len(pl)):
        pp = pl[i]
        lene = len(pp)
        lene = int(lene / 10)
        payload_count[lene] += 1
        if len(pp) >= single_packet_length:
            info_tuple = (str(date_info + ':' + flow_name), pp[:single_packet_length])
        else:
            j = i + 1
            while j < len(pl):
                pp += pl[j]
                if len(pp) >= single_packet_length:
                    info_tuple = (str(date_info + ':' + flow_name), pp[:single_packet_length])
                    break
                j += 1

            if len(pp) < single_packet_length:
                info_tuple = (str(date_info + ':' + flow_name), pp[:single_packet_length])

        data.append(info_tuple)


    # # just concat
    # pp = payload.replace(',', '')
    # while True:
    #     info_tuple = (str(date_info + ':' + flow_name), pp[:single_packet_length])
    #     data.append(info_tuple)
    #     if len(pp) < single_packet_length:
    #         break
    #     else:
    #         pp = pp[single_packet_length:]
    #
    with open(csv_file_name, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for info in data:
            writer.writerow(info)
    return payload_count


def write_date_ndpi_to_csv(date, ndpi_result_list):
    csv_file_name = os.path.join(root_dir, 'ndpi_result.csv')
    with open(csv_file_name, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for flow_name, ms_pro, app_pro, process in ndpi_result_list:
            writer.writerow((str(date + ':' + flow_name), ms_pro, app_pro, process))


if __name__ == '__main__':
    print("start to format data")
    for sub_dir in sub_dir_list():
        print("entering dir: ", sub_dir)
        file_dir = os.path.join(root_dir, sub_dir)
        ndpi_file = os.path.join(file_dir, ndpi_label_filename)
        ndpi_result = read_ndpi_result(ndpi_file)
        print("ndpi_result read")

        error_file = []
        com_result = []
        pro_result = []
        count_result = Counter()
        packet_length_counter = Counter()

        payload_length = 0
        packet_count = 0
        for flow, ms_p, app_p in ndpi_result:
            loads, process_name, count, result = read_raw_data(os.path.join(file_dir, flow))
            if result:
                packet_length_counter = write_payload_to_csv(sub_dir, flow, loads, packet_length_counter)
                payload_length += len(loads)
                packet_count += count
                com_result.append((flow, ms_p, app_p, process_name))

                if (ms_p, app_p) not in pro_result:
                    pro_result.append((ms_p, app_p))
                    print("add new protocol: ", (ms_p, app_p))
                if (ms_p, app_p) in pro_result:
                    count_result[(ms_p, app_p)] += 1
            else:
                # mark the file-missing item
                error_file.append(flow)

        print("date ", sub_dir, ": payload data writing finished")
        write_date_ndpi_to_csv(sub_dir, com_result)
        print("date ", sub_dir, ": label writing finished")
        print("\t total packet: ", packet_count, ", total bytes:", payload_length, "bytes")
        print("\t total session counted: ", len(com_result))
        print("\t total protocol counted: ", len(count_result))
        print("\t ", count_result)
        print("\t packet length distribute: ", packet_length_counter)
        print("\t missing total ", len(error_file), " files:")

        # with open(os.path.join(root_dir, 'length_dis.csv'), 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for key, value in packet_length_counter.items():
        #         writer.writerow(list(key) + [value])

        # for err_info in error_file:
        #     print("\t\t", err_info)
        print("---------------------------------------------------------")
