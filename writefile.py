import os

FILE_PATH = "results/test.txt"

if not os.path.exists(FILE_PATH):
    print("creating results file")
    with open(FILE_PATH, mode="w") as file: 
        file.write("experiment,acc,prec,rec,f1\n")

last_experiment_idx = 1
with open(FILE_PATH, mode='r') as file: 
    last_experiment_idx = len(file.readlines())

with open(FILE_PATH, mode='a') as file:
    file.write(f"{last_experiment_idx},{1},{1},{1},{1}\n")