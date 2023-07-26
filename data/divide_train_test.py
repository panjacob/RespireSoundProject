import csv
import shutil
import pandas as pd
if __name__ == "__main__":
    train_test_list = pd.read_csv('ICBHI_challenge_train_test.txt', sep="\t", names=['file', 'set'])
    train_files = train_test_list[train_test_list['set']=='train']['file']
    test_files = train_test_list[train_test_list['set']=='test']['file']
    print("moving train files..")
    for file in train_files:
        [shutil.move("./official/" + file + ext, "./official/train/" + file + ext) for ext in ['.txt', '.wav']]
    print("moving test files..")
    for file in test_files:
        [shutil.move("./official/" + file + ext, "./official/test/" + file + ext) for ext in ['.txt', '.wav']]
    print("done..")

