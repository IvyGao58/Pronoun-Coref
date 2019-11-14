import os

import jsonlines
from sklearn.model_selection import KFold


def main(*path):
    # 读入全部jsonlines文件
    data_set = []
    for i in range(len(path)):
        with jsonlines.open(path[i], mode='r') as reader:
            for item in reader:
                data_set.append(item)
        reader.close()

    # 分成10个文件
    kf = KFold(n_splits=5, shuffle=True)
    count = 0
    for train_index, test_index in kf.split(data_set):
        count += 1
        print("TRAIN:", train_index, "TEST:", test_index)
        train_set = []
        test_set = []
        for index in train_index:
            train_set.append(data_set[index])
        for index in test_index:
            test_set.append(data_set[index])
        file_name = ['train_', 'dev_']

        for name in file_name:
            cur_path = os.path.join('data/law/5_fold', name + str(count) + '.jsonlines')
            target_set = train_set if name == 'train_' else test_set
            with jsonlines.open(cur_path, mode='w') as f:
                for item in target_set:
                    f.write(item)
            f.close()


if __name__ == '__main__':
    main('data/law/train.jsonlines', 'data/law/dev.jsonlines')
