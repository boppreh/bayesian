import sys

sys.path.append('../')

import unittest
from bayesian import Bayes, classify

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

    def test_get_odds(self):
        b = Bayes({'a': 10, 'b': 50})
        self.assertEqual(b['a'],   10)
        self.assertEqual(b['b'],   50)
        self.assertEqual(b[0], 10)
        self.assertEqual(b[1], 50)

        with self.assertRaises(IndexError):
            b[2]

        with self.assertRaises(ValueError):
            b['c']

    def test_set_odds(self):
        b = Bayes((10, 20, 30))
        b[0] = 50
        b[1] = 40
        b[2] = 30
        self.assertEqual(b, [50, 40, 30])

    def test_opposite(self):
        b = Bayes([0.2, 0.8])
        opposite = b.opposite()
        self.assertEqual(opposite[0] / opposite[1], b[1] / b[0])

        b = Bayes([0.2, 0.4, 0.4])
        opposite = b.opposite()
        self.assertEqual(opposite[0] / opposite[1], b[1] / b[0])
        self.assertEqual(opposite[1] / opposite[2], b[2] / b[1])
        self.assertEqual(opposite[0] / opposite[2], b[2] / b[0])

    def test_normalized(self):
        self.assertEqual(Bayes([]).normalized(), [])
        self.assertEqual(Bayes([2]).normalized(), [1])
        self.assertEqual(Bayes([9, 1]).normalized(), [0.9, 0.1])
        self.assertEqual(Bayes([2, 4, 4]).normalized(), [0.2, 0.4, 0.4])
        self.assertEqual(Bayes([2, 0]).normalized(), [1.0, 0])
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(Bayes([0, 0]).normalized(), [1.0, 0])

    def test_operators(self):
        b = Bayes([5, 2, 3])
        b *= (2, 2, 1)
        b /= (2, 2, 1)
        self.assertEqual(b, [5, 2, 3])

        self.assertEqual(Bayes([.5, .5]) * (.9, .1), [0.45, 0.05])
        self.assertEqual(Bayes([.5, .5]) / (.9, .1), [5 / 9, 5])

        self.assertEqual(Bayes([.5, .5]) * {'0': 0.9, '1': 0.1}, [0.45, 0.05])
        self.assertEqual(Bayes([.5, .5]) * [('0', 0.9), ('1', 0.1)], [0.45, 0.05])

    def test_equality(self):
        b1 = Bayes([0.5, 0.2, 0.3])
        b2 = Bayes([5, 2, 3])
        b3 = Bayes([5, 2, 5])
        self.assertEqual(b1, b2)
        self.assertNotEqual(b1, b3)
        self.assertNotEqual(b2, b3)

    def test_update(self):
        b = Bayes([1, 2])
        b.update((2, 1))
        self.assertEqual(b, [1, 1])
        b.update((2, 1))
        self.assertEqual(b, [2, 1])
        b.update((2, 0))
        self.assertEqual(b, [1, 0])

    def test_update_from_events(self):
        b = Bayes([1, 1])
        b.update_from_events(['a', 'a', 'a'], {'a': (0.5, 2)})
        self.assertEqual(b, [0.5 ** 3, 2 ** 3])

    def test_update_from_tests(self):
        b = Bayes([1, 1])
        b.update_from_tests([True], [0.9, 0.1])
        self.assertEqual(b, [0.45, 0.05])

        b = Bayes([1, 1])
        b.update_from_tests([True, True, True, False], [0.5, 2])
        self.assertEqual(b, [0.5 ** 2, 2 ** 2])

    def test_most_likely(self):
        b = Bayes({'a': 9, 'b': 1})
        self.assertEqual(b.most_likely(), 'a')
        self.assertEqual(b.most_likely(0), 'a')
        self.assertEqual(b.most_likely(0.89), 'a')
        self.assertIsNone(b.most_likely(0.91))

    def test_is_likely(self):
        b = Bayes({'a': 9, 'b': 1})
        self.assertTrue(b.is_likely('a'))
        self.assertTrue(b.is_likely('a', 0.89))
        self.assertFalse(b.is_likely('a', 0.91))

    def test_conversions(self):
        b = Bayes({'a': 9, 'b': 1, 'c': 0})
        self.assertEqual(b, b.normalized())
        self.assertEqual(b.normalized()['a'], 0.9)
        self.assertEqual(b.opposite().opposite(), b)

    def test_extract_events_odds(self):
        instances = {'spam': ["buy viagra", "buy cialis"] * 100 + ["meeting love"],
                     'genuine': ["meeting tomorrow", "buy milk"] * 100}
        odds = Bayes.extract_events_odds(instances)

        b = Bayes({'spam': 0.9, 'genuine': 0.1})
        b.update_from_events('buy coffee for meeting'.split(), odds)
        self.assertEqual(b.most_likely(0.8), 'genuine')


class TestClassify(unittest.TestCase):
    def test_single(self):
        self.assertEqual(classify('a', {'A': []}), 'A')
        self.assertEqual(classify('a', {'A': ['a']}), 'A')
        self.assertEqual(classify('a', {'A': ['a', 'a']}), 'A')
        self.assertEqual(classify('a', {'A': ['a', 'b']}), 'A')

if __name__ == '__main__':
    unittest.main()
