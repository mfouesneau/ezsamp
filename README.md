SAMPy Client package
====================

This class aimed at providing a very small VO interactivity using the SAMP
protocol. This allows anyone to easily send and receive data to VO applications
such as Aladin or Topcat.

It provides 2 Classes:

* Hub:
    Samp hub that is required to manage the communications between all the VO applications

* Client:
    Python object that is a proxy to send and receive data from/to applications

Requirements
------------
* astropy:
    provides a samp access (astropy.vo.samp) for both python 2 and 3
    refactored version of sampy

    provides a replacement to pyfits


Optional dependencies
---------------------
* eztables:
    module that provides table manipulations regardless of storage format
    https://github.com/mfouesneau/eztables

    may not work with python 3. Automatic fallback to an internal simplified version


Example
-------
```python
import numpy
c  = Client()

# Some data is generated from my program. Let's use some mock data.
x = numpy.arange(0, 100)
y = x ** 2

# broadcast a table to topcat
c['t0'] = {'x':x, 'y':y }

# if you have a VO app that broadcasted a table 'test'
t = c['test']
```
