import sys
import numpy as np

def read_input(file):
    """
    生成器，避免一次载入全部数据至内存中
    """
    for l in file:
        yield l.rstrip()

input = read_input(sys.stdin)  #  从标准输入中读取数据
mapperout = [line.split('\t') for line in input]
cum_val = 0.0
cum_sum_sq = 0.0
cumN = 0.0
for instance in mapperout:
    nj = float(instance[0])
    cumN += nj
    cum_val += nj*float(instance[1])
    cum_sum_sq += nj*float(instance[2])
mean = cum_val/cumN
var_sum = cum_sum_sq/cumN
print("%d\t%f\t%f" % (cumN, mean, var_sum))
print("report: still alive", file=sys.stderr)

