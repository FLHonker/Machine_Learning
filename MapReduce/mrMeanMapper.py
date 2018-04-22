import sys
import numpy as np

def read_input(file):
    """
    生成器，避免一次载入全部数据至内存中
    """
    for l in file:
        yield l.rstrip()

input = read_input(sys.stdin)  #  从标准输入中读取数据
input = [float(line) for line in input]
num_Inputs = len(input)
input = np.mat(input)
sq_input = np.power(input, 2)
print("%d\t%f\t%f" % (num_Inputs, np.mean(input), np.mean(sq_input)))
print("report: still alive", file=sys.stderr)

