import knor
import numpy as np

FN = "../../test-data/matrix_r50_c5_rrw.bin"

def dim_test_c_comp():
    data = np.random.random((10,3))
    return knor.Pykmeans(data, 4)

def dim_test_c_im():
    data = np.random.random((10,3))
    centers = np.random.random((3,3))
    return knor.Pykmeans(data, centers)

def dexm_test_c_comp():
    centers = 8
    return knor.Pykmeans2(FN, centers, nrow=50, ncol=5)

def dexm_test_c_im():
    centers = np.random.random((2,5))
    return knor.Pykmeans2(FN, centers, nrow=50, ncol=5)

def test_err():
    return knor.Pykmeans2(FN, 7.2, nrow=50, ncol=5)



print "\n\n***************TEST******************\n\n"
print dim_test_c_comp()
print "\n\n***************TEST******************\n\n"
print dim_test_c_im()
print "\n\n***************TEST******************\n\n"
print dexm_test_c_comp()
print "\n\n***************TEST******************\n\n"
print dexm_test_c_im()
print "\n\n***************TEST******************\n\n"
print test_err()
