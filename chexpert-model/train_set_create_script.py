from create_semi_supervised_train_set import filter_train_set
from constants import *

if __name__ == "__main__":
    for i in [10, 100, 500, 1000]:
        for task in LOCALIZATION_TASKS:
            print("FILTERING TASK: ", task)
            filter_train_set(task, i)
