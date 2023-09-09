# -*- coding: utf-8 -*-
#
# This file is part of intbitset.
# Copyright (C) 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016 CERN.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# intbitset is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# intbitset is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with intbitset; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.

import copy
import pickle
import zlib
from typing import NamedTuple

import pytest

from bitcode.bitcode import intbitset


def b(s):
    # carried from six
    return s.encode("latin-1")


# def check_bitset(intbitset1, msg=""):
#     """
#     Run basic sanity checks on a bitset
#     """
#     wordbitsize = intbitset1.get_wordbitsize()
#     size1 = intbitset1.get_size()
#     allocated1 = intbitset1.get_allocated()
#     creator_list = intbitset1.extract_finite_list()
#     up_to1 = creator_list and max(creator_list) or -1
#     assert up_to1 <= size1 * wordbitsize < allocated1 * wordbitsize, (
#         "up_to1=%s, size1=%s, allocated1=%s while testing %s during %s"
#         % (up_to1, size1 * wordbitsize, allocated1 * wordbitsize, intbitset1, msg),
#     )
#     tmp = intbitset(intbitset1.fastdump())
#     size2 = tmp.get_size()
#     allocated2 = tmp.get_allocated()
#     creator_list = tmp.extract_finite_list()
#     up_to2 = creator_list and max(creator_list) or -1
#     assert up_to2 <= size2 * wordbitsize < allocated2 * wordbitsize, (
#         "After serialization up_to2=%s, size2=%s, allocated2=%s while testing %s during %s"
#         % (up_to2, size2 * wordbitsize, allocated2 * wordbitsize, intbitset1, msg),
#     )


TEST_SETS = [
    [1024],
    [10, 20],
    [10, 40],
    [60, 70],
    [60, 80],
    [10, 20, 60, 70],
    [10, 40, 60, 80],
    [1000],
    [10000],
    [23, 45, 67, 89, 110, 130, 174, 1002, 2132, 23434],
    [700, 2000],
    list(range(1000, 1100)),
    [30],
    [31],
    [32],
    [33],
    [62],
    [63],
    [64],
    [65],
    [126],
    [127],
    [128],
    [129],
]


class Function(NamedTuple):
    intbitset_function: str
    set_function: str
    int_function: str
    label: str
    inplace: bool


AND_FUNC = Function(
    intbitset_function=intbitset.__and__,  # NOQA
    set_function=set.__and__,
    int_function=int.__and__,
    label="intersection",
    inplace=False,
)

IAND_FUNC = Function(
    intbitset_function=intbitset.__iand__,  # NOQA
    set_function=set.__iand__,
    int_function=int.__and__,
    label="intersection_update",
    inplace=True,
)

OR_FUNC = Function(
    intbitset_function=intbitset.__or__,  # NOQA
    set_function=set.__or__,
    int_function=int.__or__,
    label="union",
    inplace=False,
)

IOR_FUNC = Function(
    intbitset_function=intbitset.__ior__,  # NOQA
    set_function=set.__ior__,
    int_function=int.__or__,
    label="union_update",
    inplace=True,
)

XOR_FUNC = Function(
    intbitset_function=intbitset.__xor__,  # NOQA
    set_function=set.__xor__,
    int_function=int.__xor__,
    label="symmetric_difference",
    inplace=False,
)

IXOR_FUNC = Function(
    intbitset_function=intbitset.__ixor__,  # NOQA
    set_function=set.__ixor__,
    int_function=int.__xor__,
    label="symmetric_difference_update",
    inplace=True,
)

SUB_FUNC = Function(
    intbitset_function=intbitset.__sub__,  # NOQA
    set_function=set.__sub__,
    int_function=int.__sub__,
    label="difference",
    inplace=False,
)

ISUB_FUNC = Function(
    intbitset_function=intbitset.__isub__,  # NOQA
    set_function=set.__isub__,
    int_function=int.__sub__,
    label="difference_update",
    inplace=True,
)

FUNCTIONS = [
    SUB_FUNC,
    ISUB_FUNC,
    AND_FUNC,
    IAND_FUNC,
    OR_FUNC,
    IOR_FUNC,
    XOR_FUNC,
    IXOR_FUNC,
]


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_list_dump(set1):
    """intbitset - list dump"""
    assert set1 == list(intbitset(set1))


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_ascii_bit_dump(set1):
    """intbitset - ascii bit dump"""
    tot = 0
    count = 0
    for bit in intbitset(set1).strbits():
        if bit == "0":
            assert count not in set1
        elif bit == "1":
            assert count in set1
            tot += 1
        else:
            raise Exception()
        count += 1
    assert len(set1) == tot


# @pytest.mark.parametrize(
#     argnames="set1",
#     argvalues=TEST_SETS + [[]],
# )
# def test_tuple_of_tuples(set1):
#     """intbitset - support tuple of tuples"""
#     tmp_tuple = tuple([(elem,) for elem in set1])
#     assert list(intbitset(set1)) == list(intbitset(tmp_tuple))
#
#
# @pytest.mark.parametrize(
#     argnames="set1",
#     argvalues=TEST_SETS + [[]],
# )
# def test_tuple_of_tuples_with_trailing_bits(set1):
#     tmp_tuple = tuple([(elem,) for elem in set1])
#     assert intbitset(set1, trailing_bits=True) == intbitset(tmp_tuple, trailing_bits=True)


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
@pytest.mark.parametrize(
    argnames="trailing_bits",
    argvalues=[True, False],
)
def test_pickling(set1, trailing_bits):
    assert intbitset(set1, trailing_bits=trailing_bits) == pickle.loads(
        pickle.dumps(intbitset(set1, trailing_bits=trailing_bits), -1)
    )


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_emptiness(set1):
    assert (not set(set1)) == (not intbitset(set1))


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_len(set1):
    intbitset1 = intbitset(set1)
    pythonset1 = set(set1)
    assert len(pythonset1) == len(intbitset1)
    intbitset1.add(76543)
    pythonset1.add(76543)
    assert len(pythonset1) == len(intbitset1)
    intbitset1.remove(76543)
    pythonset1.remove(76543)
    assert len(pythonset1) == len(intbitset1)


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_clear(set1):
    intbitset1 = intbitset(set1)
    intbitset1.clear()
    assert list(intbitset1) == []
    intbitset1 = intbitset(set1, trailing_bits=True)
    intbitset1.clear()
    assert list(intbitset1) == []


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
@pytest.mark.parametrize(
    argnames="set2",
    argvalues=TEST_SETS + [[]],
)
@pytest.mark.parametrize(
    argnames=["intbitset_op", "set_op"],
    argvalues=[
        (intbitset.__eq__, set.__eq__),  # NOQA
        (intbitset.__ge__, set.__ge__),  # NOQA
        (intbitset.__gt__, set.__gt__),  # NOQA
        (intbitset.__le__, set.__le__),  # NOQA
        (intbitset.__lt__, set.__lt__),  # NOQA
        (intbitset.__ne__, set.__ne__),  # NOQA
    ],
)
def test_intbitset_behaves_same_as_set_cmp(set1, set2, intbitset_op, set_op):
    """intbitset - (non infinite) set comparison"""
    expected = set_op(set(set1), set(set2))
    result = intbitset_op(intbitset(set1), intbitset(set2))
    assert expected == result


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
@pytest.mark.parametrize(
    argnames="trailing_bits",
    argvalues=[True, False],
)
def test_set_cloning(set1, trailing_bits):
    intbitset1 = intbitset(set1, trailing_bits=trailing_bits)
    intbitset2 = intbitset(intbitset1)
    intbitset3 = copy.deepcopy(intbitset2)
    # check_bitset(intbitset1)
    # check_bitset(intbitset2)
    # check_bitset(intbitset3)
    assert intbitset2 == intbitset1
    assert intbitset3 == intbitset1


def test_set_isdisjoint():
    sets = [
        intbitset(set([1, 2])),
        intbitset(set([3, 4])),
        intbitset(set([2, 3])),
    ]

    for set1 in sets:
        for set2 in sets:
            if set1 is not set2:
                print(set1.isdisjoint(set2), set(set1).isdisjoint(set(set2)))
                assert set1.isdisjoint(set2) is set(set1).isdisjoint(set(set2))


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_pop(set1):
    intbitset1 = intbitset(set1)
    pythonlist1 = list(set1)
    while True:
        try:
            res1 = pythonlist1.pop()
        except IndexError:
            with pytest.raises(KeyError):  # NOQA
                intbitset1.pop()
            # check_bitset(intbitset1)
            break
        res2 = intbitset1.pop()
        # check_bitset(intbitset1)
        assert res2 == res1


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_getitem(set1):
    intbitset1 = intbitset(set1)
    pythonlist1 = list(set1)
    for i in range(-2 * len(set1) - 2, 2 * len(set1) + 2):
        try:
            res1 = pythonlist1[i]
        except IndexError:
            with pytest.raises(IndexError):  # NOQA
                intbitset1.__getitem__(i)
            continue

        assert intbitset1[i] == res1


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_getitem_slices(set1):
    intbitset1 = intbitset(set1)
    pythonlist1 = list(set1)
    for start in range(-2 * len(set1) - 2, 2 * len(set1) + 2):
        for stop in range(-2 * len(set1) - 2, 2 * len(set1) + 2):
            for step in range(1, 3):
                res1 = pythonlist1[start:stop:step]
                res2 = intbitset1[start:stop:step]
                assert list(res2) == res1, (
                        f"Failure: set={set1}, start={start}, stop={stop}, "
                        f"step={step}, found={list(res2)}, "
                        f"expected={res1}, "
                        + "indices: "
                        + str(slice(start, stop, step).indices(len(pythonlist1)))
                )


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_iterator(set1):
    print(set1)
    intbitset1 = intbitset(set1)
    # check_bitset(intbitset1)
    tmp_set1 = []
    for recid in intbitset1:
        # check_bitset(intbitset1)
        tmp_set1.append(recid)
    # check_bitset(intbitset1)
    assert tmp_set1 == set1


@pytest.mark.parametrize(
    argnames="set1",
    argvalues=TEST_SETS + [[]],
)
def test_set_iterator2(set1):
    tmp_set1 = []
    for recid in intbitset(set1):
        tmp_set1.append(recid)
    assert set1 == tmp_set1


def test_empty_generator():
    intbitset(range(0))
    intbitset(i for i in range(0))


def test_do_not_allow_removal_of_none():
    with pytest.raises(TypeError):  # NOQA
        intbitset([1, 2, 3]) - None