import time

import numpy as np

start = time.perf_counter()
a = np.array([[1, 2], [3, 4]])
a = a[:, 1]
end = time.perf_counter()
print("Numpy: ", (end - start), "Result", a)

start = time.perf_counter()
a = [[1, 2], [3, 4]]
cols = []
for r in a:
    cols.append(r[1])
end = time.perf_counter()
print("Python: ", (end - start), "Result", cols)
