#
# Validation algorithm
#
import train
import genetic
import sys
import operator


class Validater:
    def __init__(self, expression):
        self.expression = expression

    def validate(self, dataset):
        func = genetic.toolbox.compile(expr=self.expression)

        for input_class in dataset.input_data:
            method_rank = []
            for method in input_class:
                method.score = func(*method.method_data)
                method_rank.append(method)
            method_rank.sort(key=operator.attrgetter('score'))
            fault_method = 0
            for i in range(len(method_rank)):
                if method_rank[i].fault == True:
                    fault_method = i
            print(fault_method)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage : python3 validate.py model.txt data1.csv data2.csv ...\n")
        sys.exit(9)
    f = open(sys.argv[1], "r")
    expression = f.read()
    data_set = train.DataSet()
    validater = Validater(expression)
    f.close()
    for file in sys.argv[2:]:
        if not file.endswith(".csv"):
            sys.stderr.write("input file should be .csv\n")
            sys.exit(9)
        data_set.parse(file)
    validater.validate(data_set)


if __name__ == "__main__":
    main()
