# Copyright (c) nexB Inc. and others. All rights reserved.
# bitcode is a trademark of nexB Inc.

# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/bitcode for support or download.

# See https://aboutcode.org for more information about nexB OSS projects.

CFG_INTBITSET_ENABLE_SANITY_CHECKS = True


class intbitset:
    def __init__(self, rhs=None, preallocate=-1, trailing_bits=0, sanity_checks=CFG_INTBITSET_ENABLE_SANITY_CHECKS,
                 no_allocate=0):
        self.bitset = 0

        if isinstance(rhs, int):
            self.bitset = rhs
        elif isinstance(rhs, intbitset):
            self.bitset = rhs.bitset
        elif isinstance(rhs, (list, set, frozenset)):
            for value in rhs:
                self.add(value)

        self.preallocate = preallocate
        self.trailing_bits = trailing_bits
        self.sanity_checks = sanity_checks
        self.no_allocate = no_allocate

    def add(self, value):
        """
        Add an element to a set.
                This has no effect if the element is already present.
        """
        if value < 0:
            raise ValueError("Value can't be negative")
        self.bitset |= 1 << value

    def clear(self):
        self.bitset = 0

    def copy(self):
        """ Return a shallow copy of a set. """
        new = intbitset()
        intbitset.bitset = self.bitset
        return new

    def difference(self, other):
        """ Return a new intbitset with elements from the intbitset that are not in the others. """
        new = intbitset()
        new.bitset = (self.bitset ^ other.bitset) & self.bitset
        return new

    def difference_update(self, other):
        """ Update the intbitset, removing elements found in others. """
        self.bitset &= (self.bitset ^ other.bitset)

    def discard(self, value):
        """
        Remove an element from a intbitset if it is a member.
                If the element is not a member, do nothing.
        """
        self.bitset &= ~(1 << value)

    def isdisjoint(self, other):
        """ Return True if two intbitsets have a null intersection. """
        return self.intersection(*[other]).bitset == 0

    def issuperset(self, other):
        """ Report whether this set contains another set. """
        return (self.bitset & other.bitset) == other.bitset

    def issubset(self, other):
        """ Report whether another set contains this set. """
        return (self.bitset & other.bitset) == self.bitset

    def remove(self, key):
        """
        Remove an element from a set; it must be a member.
                If the element is not a member, raise a KeyError.
        """
        initial_bitset = self.bitset
        self.discard(key)
        if initial_bitset != self.bitset:
            raise KeyError(f"{key} not in bitset")

    def strbits(self):
        """
        Return a string of 0s and 1s representing the content in memory
                of the intbitset.
        """
        return bin(self.bitset)[2:]

    def symmetric_difference(self, other):
        """
        Return the symmetric difference of two sets as a new set.
                (i.e. all elements that are in exactly one of the sets.)
        """
        new = intbitset()
        new.bitset = other.bitset ^ self.bitset
        return new

    def symmetric_difference_update(self, other):
        """ Update an intbitset with the symmetric difference of itself and another. """
        self.bitset ^= other.bitset

    def tolist(self):
        """
        Legacy method to retrieve a list of all the elements inside an
                intbitset.
        """
        elements = []
        for element in self:
            elements = [element] + elements
        return elements

    def union(self, *args):
        """ Return a new intbitset with elements from the intbitset and all others. """
        new = intbitset()
        bitset = self.bitset
        for other in args:
            bitset |= other.bitset
        new.bitset = bitset
        return new

    def union_update(self, *args):  # real signature unknown
        """ Update the intbitset, adding elements from all others. """
        for other in args:
            self.bitset |= other.bitset

    def intersection(self, *args):
        """ Return a new intbitset with elements common to the intbitset and all others. """
        new = intbitset()
        bitset = self.bitset
        for other in args:
            bitset &= other.bitset
        new.bitset = bitset
        return new

    def intersection_update(self, *args):
        """ Update the intbitset, keeping only elements found in it and all others. """
        for other in args:
            self.bitset &= other.bitset

    def update(self, *args, **kwargs):
        """ Update the intbitset, adding elements from all others. """
        pass

    def __and__(self, other):
        """
            Return the intersection of two intbitsets as a new set.
            (i.e. all elements that are in both intbitsets.)
        """
        new = intbitset()
        new.bitset = self.bitset & other.bitset
        return new

    def __eq__(self, other):
        """ Return self==value. """
        return self.bitset == other.bitset

    def __contains__(self, key):
        """ Return key in self. """
        key_bit = 1 << key
        return key_bit & self.bitset == key_bit

    def __len__(self):
        """ Return len(self). """
        return len(bin(self.bitset)) - 2

    def __iter__(self):
        """ Implement iter(self). """
        bits = bin(self.bitset)[2:]
        size = len(bits)
        for bit in bits:
            if bit == "1":
                yield size
            size -= 1

    def __hash__(self):
        return self.bitset

    def __str__(self):
        binary = bin(self.bitset)[2:]
        n = len(binary)
        ans = "intbitset(["
        for char in binary:
            if char == "1":
                ans += str(n - 1)
                if n > 0:
                    ans += ", "
            n -= 1
        ans = ans.rstrip(', ')
        ans += "])"
        return ans
