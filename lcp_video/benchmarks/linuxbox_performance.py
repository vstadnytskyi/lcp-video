"""
A series of tests to examing the performance difference between two systems
"""

def sum(lst = [1e3,1e4,1e5,1e6]):
    from time import time
    import timeit
    import numpy as np
    mean = []
    std = []
    size = []
    for length in lst:
        preparation= ''
        preparation += "import numpy as np \n"
        preparation += f"arr = np.random.rand({int(length)}) \n"
        testcode = 'ttttt = arr.sum()'
        t = timeit.Timer(testcode,preparation)
        result = np.array(t.repeat(4,20))/20
        mean.append(round(result.mean(),4))
        std.append(round(result.std(),4))
        size.append(length)
    return np.array(mean), np.array(std), np.array(size)
