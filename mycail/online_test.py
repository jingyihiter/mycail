import json
from predictor import Predictor

def test(filename):
    """
    用于测试提交的预测函数
    :param filename:
    :return:
    """
    result = []
    pre = Predictor()
    with open(filename, "r", encoding='utf-8') as f:
        content = []
        num = 0
        for line in f:
            num += 1
            sample = json.loads(line.strip("\n"))
            content.append(sample['fact'])
            if num%64==0:
                result += pre.predict(content)
                content = []
    print(len(result),result[0])
test("data_128_lines.json")


