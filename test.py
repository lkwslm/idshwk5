from sklearn.ensemble import RandomForestClassifier
import math

def get_numbers(s):
    num = 0
    for i in s:
        if i.isdigit():
            num = num + 1
    return num

def get_entropy(s):
    entropy = 0
    times = {}
    for i in s:
        times[i] = s.count(i)
        entropy -= (times[i] / len(s)) * math.log(times[i] / len(s))
    return entropy

domainList = []
clf = RandomForestClassifier(random_state=0)

class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.length = len(_name)
        self.numbers = get_numbers(_name)
        self.entropy = get_entropy(_name)

    def return_data(self):
        return [self.length, self.numbers, self.entropy]

    def return_label(self):
        if self.label == 'dga':
            return 1
        else:
            return 0

def init(file):
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            tokens = line.split(',')
            name = tokens[0]
            label = tokens[1]
            domainList.append(Domain(name, label))

def predict(test_file, result_file):
    with open(test_file) as f_test, open(result_file, 'w') as f_result:
        for line in f_test:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            result = clf.predict([[len(line), get_numbers(line), get_entropy(line)]])
            if result == [0]:
                f_result.write(line + ',notdga\n')
            else:
                f_result.write(line + ',dga\n')

def main():
    init('train.txt')
    featureMatrix = []
    labelList = []
    for i in domainList:
        featureMatrix.append(i.return_data())
        labelList.append(i.return_label())
    clf.fit(featureMatrix, labelList)
    predict('test.txt', 'result.txt')

if __name__ == '__main__':
    main()