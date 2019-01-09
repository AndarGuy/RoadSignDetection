i_start = 0.3
sum = 0
c = 0.135
rt = 777.9

for i in range(1, 3001):
    print((i_start + 0.015 * i/1000), i/1000)
    sum += (i_start + 0.015 * i/1000) * 220 * 0.001 * 0.9

print((204.93 / (c * 1000 * rt)) * 1000)