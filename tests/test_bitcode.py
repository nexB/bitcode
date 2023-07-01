#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/ahocode for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

import unittest

from bitcode.bitcode import intbitset


class TestBitsetMethods(unittest.TestCase):

    def setUp(self):
        self.values = [5, 4, 2, 1, 0]
        self.set_values = (5, 4, 2, 1, 0)
        self.A = intbitset(self.values)

    def test_initialization(self):
        self.assertEqual(self.A.bitset, 55)

    def test_add(self):
        A = self.A
        A.add(10)
        A.add(3)
        A.add(0)
        A.add(1)
        expected = [5, 4, 2, 1, 0, 3, 10]
        # result_bitset = 0b10000111111
        self.assertEqual(A, intbitset(expected))

    def test_discard_and_remove(self):
        A = self.A
        A.discard(3)
        A.discard(1)
        A.discard(5)
        A.discard(23)
        A.discard(12)
        expected_after_discard = [4, 2, 0]
        # result_bitset = 0b10101
        self.assertEqual(A, intbitset(expected_after_discard))

        expected_after_remove = [0]
        A.remove(4)
        A.remove(2)
        self.assertEqual(A, intbitset(expected_after_remove))

        self.assertRaises(KeyError, A.remove, 32)

    def test_clear(self):
        self.A.clear()
        # result_bitset = 0b0
        self.assertEqual(self.A, intbitset())

    def test_iter(self):
        A = self.A
        values_added = list(iter(A))
        self.assertEqual(sorted(values_added), sorted(self.values))

    def test_difference_and_difference_updated(self):
        A = self.A
        new1 = intbitset([9, 1])
        new2 = intbitset([4])
        expected = [5, 2, 0]
        x = A.difference(*[new1, new2])
        self.assertEqual(A.difference(*[new1, new2]), intbitset(expected))
        A.difference_update(*[new1, new2])
        self.assertEqual(A, intbitset(expected))

    def test_disjoint_and_superset(self):
        A = self.A
        subset1 = intbitset([0, 5, 1, 4])
        subset2 = intbitset([5, 4, 45, 2, 1, 0, 22])

        self.assertEqual(A.issubset(subset1), False)
        self.assertEqual(A.issubset(subset2), True)

        self.assertEqual(A.issuperset(subset1), True)
        self.assertEqual(A.issuperset(subset2), False)

    def test_strbits(self):
        A = self.A
        self.assertEqual(A.strbits(), '110111')

    def test_symmetric_difference_and_symmetric_difference_update(self):
        A = self.A
        intbitset1 = intbitset([2, 1, 4, 33, 6, 8])
        expected = intbitset([33, 6, 8, 5, 0])
        self.assertEqual(A.symmetric_difference(intbitset1), expected)
        A.symmetric_difference_update(intbitset1)
        self.assertEqual(A, expected)

    def test_union_and_union_update(self):
        A = self.A
        intbitset1 = intbitset([100, 200, 300])
        intbitset2 = intbitset([68, 1, 5, 4])
        intbitset3 = intbitset([1, 2, 3])

        expected = intbitset([5, 4, 2, 1, 0, 3, 68, 100, 200, 300])
        self.assertEqual(A.union(*[intbitset1, intbitset2, intbitset3]), expected)

        A.union_update(*[intbitset1, intbitset2, intbitset3])
        self.assertEqual(A, expected)

    def test_intersection_and_intersection_update(self):
        A = self.A
        intbitset1 = intbitset([100, 200, 300, 1])
        intbitset2 = intbitset([68, 1, 5, 4])
        intbitset3 = intbitset([1, 2, 3])
        expected = intbitset([1])

        self.assertEqual(A.intersection(*[intbitset1, intbitset2, intbitset3]), expected)

        A.intersection_update(*[intbitset1, intbitset2, intbitset3])
        self.assertEqual(A, expected)

    def test_to_list(self):
        A = self.A
        self.assertEqual(sorted(A.tolist()), sorted(self.values))

    def test_len(self):
        A = self.A
        self.assertEqual(len(A), len(self.values))

    def test_print(self):
        A = self.A
        self.assertEqual(A.__str__(), "intbitset([5, 4, 2, 1, 0])")

    def test_contains(self):
        A = self.A
        self.assertEqual(5 in A, True)
        self.assertEqual(3 not in A, True)
        self.assertEqual(A.__contains__(1), True)
        self.assertEqual(A.__contains__(99), False)
