from math import *


def mod(n):
    m = n;
    n = int(n)

    for num in range(1, n+1):
        if n % num == 0 and num != 1:
            if m > num:
                m = num
    return m


def bod(n):
    m = 0;
    n = int(n)

    for num in range(1, n):
        if n % num == 0 and num != 1:
            if m < num:
                m = num
    return m


for i in range(1000):
    print(i, bod(i), mod(i), bod(i) - mod(i))