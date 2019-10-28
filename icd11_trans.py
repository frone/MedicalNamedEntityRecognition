import os
import pandas as pd
from pandas import DataFrame
import xlrd
import pickle


def read_csv_file(icd_file):
    icd_11_dict = {}
    with open(icd_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            icd_11_code = line.split(",")[0].strip()
            cn_name = line.split(",")[1].strip()
            if cn_name not in icd_11_dict.keys():
                icd_11_dict[cn_name] = [icd_11_code]
            else:
                print("=" * 50)
                print(icd_11_dict[cn_name])
                print(line.split(",")[0] + " - " + line.split(",")[1])
                icd_11_dict[cn_name].append(icd_11_code)
                print("=" * 50)
    print(len(icd_11_dict))
    return icd_11_dict


def read_xlsx_file(icd_file2):
    icd_11_dict = {}
    # icd_file2 = os.path.join(cur, "data" + os.sep + "ICD-11.xlsx")
    sheet = xlrd.open_workbook(icd_file2).sheets()[0]
    invalid_cnt = 0
    for row_index in range(sheet.nrows):
        icd_11_code = sheet.cell(row_index, 0).value
        cn_name = sheet.cell(row_index, 1).value
        if sheet.cell(row_index, 2).value == "否":
            invalid_cnt += 1
            continue
        if cn_name not in icd_11_dict.keys():
            icd_11_dict[cn_name] = [icd_11_code]
        else:
            # print("=" * 50)
            # print(icd_11_dict[cn_name])
            # print(cn_name)
            icd_11_dict[cn_name].append(icd_11_code)
            # print("=" * 50)
    print("invalid count: " + str(invalid_cnt))
    return icd_11_dict


if __name__ == "__main__":
    cur = "/".join(os.path.abspath(__file__).split(os.sep)[:-1])
    print("start...")
    print(cur)
    origin_path = os.path.join(cur, "data_origin")
    train_filepath = os.path.join(cur, "data" + os.sep + "train.txt")
    icd_file = os.path.join(cur, "data" + os.sep + "ICD11_DICT.csv")
    icd_file2 = os.path.join(cur, "data" + os.sep + "ICD-11.xlsx")
    icd_pick = os.path.join(cur, "data" + os.sep + "ICD-11.pick")
    print(origin_path)
    print(train_filepath)
    print(icd_file)
    # for root, dirs, files in os.walk(origin_path):
    #     # print("=" * 20 + "root" + "=" * 20)
    #     # print(root)
    #     # print("=" * 20 + "dirs" + "=" * 20)
    #     # print(dirs)
    #     # print("=" * 20 + "files" + "=" * 20)
    #     # print(files)
    #     for file in files:
    #         filepath = os.path.join(root, file)
    #         print(filepath)

    # 将对象写入文件
    icd_11_dict = read_xlsx_file(icd_file2)
    f = open(icd_pick, "wb")
    print(len(icd_11_dict))
    pickle.dump(icd_11_dict, f)
    more_than_3_cnt = 0
    for key, value in icd_11_dict.items():
        if len(value) >= 3:
            more_than_3_cnt += 1
            print(key + " : " + str(value))
    print("重复多于3种的数量 : " + str(more_than_3_cnt))

    f = open(icd_pick, "rb")
    data = pickle.load(f)
    print(len(icd_11_dict))
    print(data)
