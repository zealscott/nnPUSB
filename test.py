import math

x = 0.3
y = 0.2
z = 0.5

x1 = z**10/(x**10 + y**10 + z**10)

x2 = math.log10(x)/(math.log10(x) + math.log10(y) +math.log10(z))

print(x1,x2)