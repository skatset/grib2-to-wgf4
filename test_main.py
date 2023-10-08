import unittest
import numpy as np
from main import prepare_grib2_data


class TestPrepareGrib2Data(unittest.TestCase):
    def test_data_with_previous_data(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        previous_data = np.array([[0.5, 1.0], [2.0, 3.0]])
        result = prepare_grib2_data(data, previous_data)
        expected_result = np.array([[0.5, 1.0], [1.0, 1.0]])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_data_without_previous_data(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = prepare_grib2_data(data, None)
        expected_result = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_data_with_nan(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        previous_data = np.array([[0.5, 1.0], [2.0, np.nan]])
        result = prepare_grib2_data(data, previous_data)
        expected_result = np.array([[0.5, -100500.0], [1.0, 4.0]])
        self.assertTrue(np.array_equal(result, expected_result))


if __name__ == '__main__':
    unittest.main()
