import random
import torch


def f():
    return 1, 2, 3, 4


# a = [(1, 2), (3, 4), (5, 6)]
# # f(*zip(*a))
# random.shuffle(a)
# print(a)

*x, y = f()
print(x)
