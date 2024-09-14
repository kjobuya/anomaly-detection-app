import unittest
import numpy as np
import numpy.testing as npt

from patchcore import PatchCore

class EuclideanDistanceTest(unittest.TestCase):
    def test_euclidean_distances(self):
        origin = np.array([(0,0), (1,1), (2,2)], dtype=np.float64)
        targets = np.array([(4,6), (7,7), (6,9)], dtype=np.float64)
        
        def euclidean_calc (origin, target):
            assert origin.shape == target.shape
            assert origin.size == 2
            
            return np.sqrt(((target[0]-origin[0])**2) + ((target[1]-origin[1]))**2)
        
        answers = []
        for origin_element in origin:
            entry = []
            for target_element in targets:
                answer = euclidean_calc(origin_element, target_element)
                entry.append(answer)
            answers.append(entry)
            
        answers = np.array(answers, dtype=np.float64)
        
        # testing if calculation is correct
        npt.assert_array_almost_equal(PatchCore.calculate_euclidean_distances(origin, targets, method="array"), answers, decimal=8)
        npt.assert_array_almost_equal(PatchCore.calculate_euclidean_distances(origin, targets, method="tensor"), answers, decimal=8)
        
        # testing equality of methods
        npt.assert_array_almost_equal(PatchCore.calculate_euclidean_distances(origin, targets, method="array"), 
                                      PatchCore.calculate_euclidean_distances(origin, targets, method="array"), 
                                      decimal=8)
        
        print("Euclidean distance test passed")

        
if __name__ == '__main__':
    unittest.main()