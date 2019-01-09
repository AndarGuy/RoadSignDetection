def nod(n1, n2):
    m = 0;
    n1 = int(n1)
    n2 = int(n2)

    for num in range(1, n1 + 1):
        if n1 % num == 0 and n2 % num == 0:
            if m < num:
                m = num
    return m


def get(s):
    t = [nod(s[0], s[1]), nod(s[0], s[2]), nod(s[0], s[3]), nod(s[1], s[2]), nod(s[1], s[3]), nod(s[2], s[3])]
    return t;


for i in range(0, 100):
    for i2 in range(0, 100):
        for i3 in range(0, 100):
            for i4 in range(0, 100):
                print(get(str(i) + str(i2) + str(i3) + str(i4)))
