#
# Unit Testing
#
import unittest
import train
import csv


class Train_Test(unittest.TestCase):
    def test_parse_Lang_1(self):
        dataset = train.DataSet()
        file = "fluccs_data/Lang_1.csv"
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            fault_list = [row[-1] for row in reader][1:]
            compare_fault_method = 0
            for i in range(len(fault_list)):
                if fault_list[i] == "1":
                    compare_fault_method = i
        dataset.parse(file)
        methods = dataset.input_data[0]
        fault_method = 0
        for i in range(len(methods)):
            if methods[i].fault:
                fault_method = i
        self.assertEqual(fault_method, compare_fault_method)

    def test_parse_Lang_20(self):
        dataset = train.DataSet()
        file = "fluccs_data/Lang_20.csv"
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            fault_list = [row[-1] for row in reader][1:]
            compare_fault_method = 0
            for i in range(len(fault_list)):
                if fault_list[i] == "1":
                    compare_fault_method = i
        dataset.parse(file)
        methods = dataset.input_data[0]
        fault_method = 0
        for i in range(len(methods)):
            if methods[i].fault:
                fault_method = i
        self.assertEqual(fault_method, compare_fault_method)


if __name__ == "__main__":
    unittest.main()