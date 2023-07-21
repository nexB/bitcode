bitcode
==================
This repo is a pure python implementation for `intbitset <https://github.com/inveniosoftware-contrib/intbitset>`_.


Installation
-------------------

Requirements
###################
* Python 3.8 or later

.. code-block:: bash

    pip install bitcode

Documentation
---------------------------
bitcode is a fallback library for intbitset, so the ``intbitset`` class and its methods
have same names and parameters.

Below listed are the implemented classes and methods.

Classes
##########

* ``intbitset``

Methods for Automaton class
###############################

* ``add``
* ``clear``
* ``copy``
* ``difference``
* ``difference_update``
* ``discard``
* ``isdisjoint``
* ``issuperset``
* ``issubset``
* ``remove``
* ``strbits``
* ``symmetric_difference``
* ``symmetric_difference_update``
* ``tolist``
* ``union``
* ``union_update``
* ``intersection``
* ``intersection_update``
* ``__and__``
* ``__eq__``
* ``__contains__``
* ``__len__``
* ``__iter__``
* ``__hash__``
* ``__str__``

For documentation please refer to: https://intbitset.readthedocs.io/en/latest/
