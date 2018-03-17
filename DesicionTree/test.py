# li = [1,[2,3]]
# def la():
#
#     li.append(1)
# la()
# print(li)
a = [1,2,3,4,4,4]
for i in a:
    b = a.copy()
    if i == 1:
        print('i==1')
        print(i, a)
        continue
    else:
        print('i != 1')
        b.remove(4)
        # print(i, a)
print(a)
from copy import deepcopy

# def fun(list):
#     print(id(list))
#     for i in list:
#         print(id(i))
#
# a = [1,2,[3,4,5]]
# b = a.copy()
# fun(a)
# print("------b-------")
# fun(b)
# a.append(1)
# print(a)
# print(b)
