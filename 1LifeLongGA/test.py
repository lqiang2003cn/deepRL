from functools import reduce

lt = (2,3,4)


ln = reduce(lambda x,y:x * y,lt)

print(ln)