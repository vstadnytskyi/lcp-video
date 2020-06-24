
def run_test_gzip():
    def test_gzip(f):
        from numpy import copy, mean, random
        return copy(f['ndarray'][random.randint(600),:,:])

    root = '/Users/femto-13/Downloads/hdf5_test/'
    from h5py import File
    f = File(root+ '2.hdf5','r')

    from time import time
    t1 = time()

    arr = test_gzip(f)
    t2 = time()
    print('non-zip:',arr.mean(),t2-t1)

    f2 = File(root+ '3.hdf5','r')

    from time import time
    t1 = time()

    arr2 = test_gzip(f2)
    t2 = time()
    print('gzip:',arr2.mean(),arr2.std(),t2-t1)

    from numpy import allclose
    print(allclose(arr,arr2))
