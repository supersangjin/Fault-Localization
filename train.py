#
# Training/learning algorithm
#
import csv
import genetic
import sys


class Method:
    def __init__(self):
        self.method_name = ""
        self.method_data = []
        self.fault = False
        self.score = 0


class DataSet:
    def __init__(self):
        self.input_data = []  # training data

    def parse(self, file):
        temp_data = []
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == "method":  # first row
                    continue
                method = Method()
                method.method_name = row[0]
                for element in row[1:-1]:
                    method.method_data.append(float(element))
                if row[-1] == "1":
                    method.fault = True
                temp_data.append(method)
        self.input_data.append(temp_data)


class Trainer:
    def __init__(self):
        self.hof = None

    def train(self, data_set):
        self.hof = genetic.train(data_set)

    def output(self):
        f = open('model.txt', 'w')
        f.write(str(self.hof))
        f.close()


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage : python3 train.py data1.csv data2.csv ...\n")
        sys.exit(9)
    trainer = Trainer()
    data_set = DataSet()
    for file in sys.argv[1:]:
        if not file.endswith(".csv"):
            sys.stderr.write("input file should be .csv\n")
            sys.exit(9)
        data_set.parse(file)
    trainer.train(data_set)
    trainer.output()


if __name__ == "__main__":
    main()
