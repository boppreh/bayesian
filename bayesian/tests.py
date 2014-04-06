import sys

sys.path.append('../')

import unittest
from bayesian import Bayes

class TestBayes(unittest.TestCase):
    def test_empty_constructor(self):
        with self.assertRaises(ValueError):
            b = Bayes()

    def test_list_constructor(self):
        self.assertEqual(Bayes([]), [])
        self.assertEqual(Bayes(()), [])
        self.assertEqual(Bayes(range(5)), [0, 1, 2, 3, 4])
        self.assertEqual(Bayes({'a': 10, 'b': 50}), [10, 50])
        self.assertEqual(Bayes([10, 10, 20]), [10, 10, 20])
        self.assertEqual(Bayes([('a', 10), ('b', 50)]), [10, 50])
        with self.assertRaises(ValueError):
            b = Bayes([('a', 10), ('b', 50), ('a', 15)])

if __name__ == '__main__':
    unittest.main()
