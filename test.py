import numpy as np  # 如果是普通列表也适用
import sys
your_array= np.random.rand(5000)


np.set_printoptions(threshold=sys.maxsize)
print(your_array)
full_str = str(your_array)
with open("full_array.txt", "w") as f:
    f.write(full_str)