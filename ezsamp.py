"""
SAMPy Client package
====================

This class aimed at providing a very small VO interactivity using the SAMP
protocol. This allows anyone to easily send and receive data to VO applications
such as Aladin or Topcat.

It provides 3 Classes:

* Hub:
    Samp hub that is required to manage the communications between all the VO applicati
ons

* Client:
    Python object that is a proxy to send and receive data from/to applications

* SimpleTable:
    a fallback class of eztables that allows users to easily manipulate tables
    (see optional dependencies)

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
see the function demo() for a usage example. (demo??)

.. code::
    import numpy
    c  = Client()

    # Some data is generated from my program. Let's use some mock data.
    x = numpy.arange(0, 100)
    y = x ** 2

    # broadcast a table to topcat
    c['t0'] = {'x':x, 'y':y }

    if client is None:
        return c
"""
from __future__ import (absolute_import, division, print_function)

__version__ = '2.0'
import atexit
import os
import sys

import numpy as np
from numpy.lib import recfunctions
from copy import deepcopy

try:
    from astropy.vo import samp as sampy
except ImportError:
    from . import sampy

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

# ================================================================
# Python 3 compatibility behavior
# ================================================================
# remap some python 2 built-ins on to py3k behavior or equivalent
# Most of them become generators
import operator

PY3 = sys.version_info[0] > 2

if PY3:
    iteritems = operator.methodcaller('items')
    itervalues = operator.methodcaller('values')
else:
    range = xrange
    from itertools import izip as zip
    iteritems = operator.methodcaller('iteritems')
    itervalues = operator.methodcaller('itervalues')


#==========================================================================
# SimpleTable -- provides table manipulations with limited storage formats
#==========================================================================
class SimpleTable(object):

    def __init__(self, fname, dtype=None, **kwargs):

        if (type(fname) == dict) or (dtype in [dict, 'dict']):
            self.header = fname.pop('header', {})
            self.data = self._convert_dict_to_structured_ndarray(fname)
        elif type(fname) in [str]:
            extension = fname.split('.')[-1]
            if (extension == 'csv') or dtype == 'csv':
                self.data = np.recfromcsv(fname, **kwargs)
                self.header = {}
            elif (extension == 'fits') or dtype == 'fits':
                self.data = np.array(pyfits.getdata(fname, **kwargs))
                self.header = pyfits.getheader(fname, ext=1, **kwargs)
            else:
                raise Exception('Format {0:s} not handled'.format(extension))
        elif type(fname) == np.ndarray:
            self.data = fname
            self.header = {}
        elif type(fname) == SimpleTable:
            self.data = fname.data
            self.header = fname.header
        else:
            raise Exception('Type {0!s:s} not handled'.format(type(fname)))
        if 'NAME' not in self.header:
            self.header['NAME'] = 'No Name'

    def write(self, fname, **kwargs):
        extension = fname.split('.')[-1]
        if (extension == 'csv'):
            np.savetxt(fname, self.data, delimiter=',', header=self.header, **kwargs)
        elif (extension in ['txt', 'dat']):
            np.savetxt(fname, self.data, delimiter=' ', header=self.header, **kwargs)
        elif (extension == 'fits'):
            pyfits.writeto(fname, self.data, self.header, **kwargs)
        else:
            raise Exception('Format {0:s} not handled'.format(extension))

    def _convert_dict_to_structured_ndarray(self, data):
        """convert_dict_to_structured_ndarray

        Parameters
        ----------

        data: dictionary like object
            data structure which provides iteritems and itervalues

        returns
        -------
        tab: structured ndarray
            structured numpy array
        """
        newdtype = []
        for key, dk in iteritems(data):
            _dk = np.asarray(dk)
            dtype = _dk.dtype
            # unknown type is converted to text
            if dtype.type == np.object_:
                if len(data) == 0:
                    longest = 0
                else:
                    longest = len(max(_dk, key=len))
                    _dk = _dk.astype('|%iS' % longest)
            if _dk.ndim > 1:
                newdtype.append((str(key), _dk.dtype, (_dk.shape[1],)))
            else:
                newdtype.append((str(key), _dk.dtype))
        tab = np.rec.fromarrays(itervalues(data), dtype=newdtype)
        return tab

    def keys(self):
        return self.colnames

    @property
    def colnames(self):
        return self.data.dtype.names

    @property
    def ncols(self):
        return len(self.colnames)

    @property
    def nrows(self):
        return len(self.data)

    @property
    def nbytes(self):
        """ return the number of bytes of the object """
        n = sum(k.nbytes if hasattr(k, 'nbytes') else sys.getsizeof(k) for k in self.__dict__.values())
        return n

    def __len__(self):
        return self.nrows

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, v):
        return np.asarray(self.data.__getitem__(v))

    def __setitem__(self, v):
        return self.data.__setitem__(v)

    def __iter__(self):
        return self.data.__iter__()

    def iterkeys(self):
        for k in self.keys():
            yield k

    def itervalues(self):
        for l in self.data:
            yield l

    def __repr__(self):
        s = object.__repr__(self)
        s += "\nTable: {name:s} nrows={s.nrows:d}, ncols={s.ncols:d}"
        return s.format(name=self.header.get('NAME', 'Noname'), s=self)

    def __getslice__(self, i, j):
        return self.data.__getslice__(i, j)

    def __contains__(self, k):
        return (k in self.keys()) or (k in self._aliases)

    def __array__(self):
        return self.data

    def sort(self, keys):
        """
        Sort the table inplace according to one or more keys. This operates on
        the existing table (and does not return a new table).

        Parameters
        ----------

        keys: str or seq(str)
            The key(s) to order by
        """
        if not hasattr(keys, '__iter__'):
            keys = [keys]
        self.data.sort(order=keys)

    def match(self, r2, key):
        """ Returns the indices at which the tables match
        matching uses 2 columns that are compared in values

        Parameters
        ----------
        r2:  Table
            second table to use

        key: str
            fields used for comparison.

        Returns
        -------
        indexes: tuple
            tuple of both indices list where the two columns match.
        """
        return np.where( np.equal.outer( self[key], r2[key] ) )

    def stack(self, r, defaults=None):
        """
        Superposes arrays fields by fields inplace

        Parameters
        ----------
        r: Table
        """
        if not hasattr(r, 'data'):
            raise AttributeError('r should be a Table object')
        self.data = recfunctions.stack_arrays( [self.data, r.data], defaults, usemask=False, asrecarray=True)

    def join_by(self, r2, key, jointype='inner', r1postfix='1', r2postfix='2',
                defaults=None, asrecarray=False, asTable=True):
        """
        Join arrays `r1` and `r2` on key `key`.

        The key should be either a string or a sequence of string corresponding
        to the fields used to join the array.
        An exception is raised if the `key` field cannot be found in the two input
        arrays.
        Neither `r1` nor `r2` should have any duplicates along `key`: the presence
        of duplicates will make the output quite unreliable. Note that duplicates
        are not looked for by the algorithm.

        Parameters
        ----------
        key: str or seq(str)
            corresponding to the fields used for comparison.

        r2: Table
            Table to join with

        jointype: str in {'inner', 'outer', 'leftouter'}
            'inner'     : returns the elements common to both r1 and r2.
            'outer'     : returns the common elements as well as the elements of r1
                          not in r2 and the elements of not in r2.
            'leftouter' : returns the common elements and the elements of r1 not in r2.

        r1postfix: str
            String appended to the names of the fields of r1 that are present in r2

        r2postfix:  str
            String appended to the names of the fields of r2 that are present in r1

        defaults:   dict
            Dictionary mapping field names to the corresponding default values.

        Returns
        -------
        tab: Table
            joined table

        .. notes::

            * The output is sorted along the key.

            * A temporary array is formed by dropping the fields not in the key
              for the two arrays and concatenating the result. This array is
              then sorted, and the common entries selected. The output is
              constructed by filling the fields with the selected entries.
              Matching is not preserved if there are some duplicates...
        """
        arr = recfunctions.join_by(key, self, r2, jointype=jointype,
                                   r1postfix=r1postfix, r2postfix=r2postfix,
                                   defaults=defaults, usemask=False,
                                   asrecarray=True)

        return SimpleTable(arr)

    @property
    def empty_row(self):
        """ Return an empty row array respecting the table format """
        return np.rec.recarray(shape=(1,), dtype=self.data.dtype)

    def append_row(self, iterable):
        """
        Append one row in this table.

        see also: :func:`stack`

        Parameters
        ----------
        iterable: iterable
            line to add
        """
        assert( len(iterable) == self.ncols ), 'Expecting as many items as columns'
        r = self.empty_row
        for k, v in enumerate(iterable):
            r[0][k] = v
        self.stack(r)

    addLine = append_row

    def remove_columns(self, names):
        """
        Remove several columns from the table

        Parameters
        ----------
        names: sequence
            A list containing the names of the columns to remove
        """
        self.pop_columns(names)

    remove_column = delCol = remove_columns

    def pop_columns(self, names):
        """
        Pop several columns from the table

        Parameters
        ----------

        names: sequence
            A list containing the names of the columns to remove

        Returns
        -------

        values: tuple
            list of columns
        """

        if not hasattr(names, '__iter__') or type(names) in [str]:
            names = [names]

        p = [self[k] for k in names]
        self.data = recfunctions.drop_fields(self.data, names)

        return p

    def find_duplicate(self, index_only=False, values_only=False):
        """Find duplication in the table entries, return a list of duplicated elements
            Only works at this time is 2 lines are *the same entry*
            not if 2 lines have *the same values*
        """
        dup = []
        idd = []
        for i in range(len(self.data)):
            if (self.data[i] in self.data[i + 1:]):
                if (self.data[i] not in dup):
                    dup.append(self.data[i])
                    idd.append(i)
        if index_only:
            return idd
        elif values_only:
            return dup
        else:
            return zip(idd, dup)

    def evalexpr(self, expr, exprvars=None, dtype=float):
        """ evaluate expression based on the data and external variables
            all np function can be used (log, exp, pi...)

        Parameters
        ----------
        expr: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        exprvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        dtype: dtype definition
            dtype of the output array

        Returns
        -------
        out : NumPy array
            array of the result
        """
        _globals = {}
        for k in ( self.keys() + self._aliases.keys() ):
            _globals[k] = self[k]

        if exprvars is not None:
            assert(hasattr(exprvars, 'keys') & hasattr(exprvars, '__getitem__' )), "Expecting a dictionary-like as condvars"
            for k, v in ( exprvars.items() ):
                _globals[k] = v

        # evaluate expression, to obtain the final filter
        r    = np.empty( self.nrows, dtype=dtype)
        r[:] = eval(expr, _globals, np.__dict__)

        return r

    def where(self, condition, condvars=None, *args, **kwargs):
        """ Read table data fulfilling the given `condition`.
        Only the rows fulfilling the `condition` are included in the result.

        Parameters
        ----------
        condition: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        condvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        Returns
        -------
        out: ndarray/ tuple of ndarrays
            result equivalent to numpy.where

        """
        ind = np.where(self.evalexpr(condition, condvars, dtype=bool ), *args, **kwargs)
        return ind

    def selectWhere(self, fields, condition, condvars=None, **kwargs):
        """ Read table data fulfilling the given `condition`.
            Only the rows fulfilling the `condition` are included in the result.

        Parameters
        ----------
        condition: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        condvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        Returns
        -------
        """
        # make a copy without the data itself (memory gentle)
        tab = self.__class__()
        for k in self.__dict__.keys():
            if k != 'data':
                setattr(tab, k, deepcopy(self.__dict__[k]))

        if fields.count(',') > 0:
            _fields = fields.split(',')
        elif fields.count(' ') > 0:
            _fields = fields.split()
        else:
            _fields = fields

        if condition in [True, 'True', None]:
            ind = None
        else:
            ind = self.where(condition, condvars, **kwargs)

        if _fields == '*':
            if ind is not None:
                tab.data = self.data[ind]
            else:
                tab.data = deepcopy(self.data)
        else:
            if ind is not None:
                tab.data = self.data[_fields][ind]
            else:
                tab.data = self.data[_fields]
            names = tab.data.dtype.names
            #cleanup aliases and columns
            for k in self.keys():
                if k not in names:
                    al = self.reverse_alias(k)
                    for alk in al:
                        tab.delCol(alk)
                    if k in tab.keys():
                        tab.delCol(k)

        comm = 'SELECT {0:s} FROM {1:s} WHERE {2:s}'
        tab.header['COMMENT'] = comm.format(','.join(_fields),
                                            self.header['NAME'],
                                            condition)
        return tab


try:
    from eztables import Table
except Exception as e:
    print('Warning: eztable could not be imported for the following reason: {0}'.format(e))
    print('Warning: switching to SimpleTable instead')
    Table = SimpleTable


#==========================================================================
# HUB -- Generate a SAMP hub to manage communications
#==========================================================================
class Hub(object):
    """
    This class is a very minimalistic class that provides a working SAMP hub

    Example
    -------
    >>> h = Hub()
        # many complex operations
        h.stop()
    """
    def __init__(self,addr, *args, **kwargs):
        self.SAMPHubServer = sampy.SAMPHubServer(addr=addr, *args, **kwargs)
        atexit.register(self.SAMPHubServer.stop)

    def __del__(self):
        self.SAMPHubServer.stop()


#==========================================================================
# Client -- Generate a SAMP client able to send and receive data
#==========================================================================
class Client(object):
    """
    This class implenent an interface to SAMP applications like Topcat and
    Aladin using Sampy module.
    It allows you to exchange tables with topcat using the SAMP system.

    To instanciate a connection:
    >>> client = Client(addr='localhost', hub=True)

    This could create a local hub to connect to a local session
    Nb: Topcat need to be running when sending messages, however this
        could be done later.

    To send a table to Topcat:
    >>> client['tblName'] = {  }  # with some content in the dict

    To receive a table from Topcat:
    (broadcast the table from Topcat)
    >>> table = client['tblName']
    """
    #Destructor ===============================================================
    def __del__(self):
        self.client.disconnect()
        if self.hub is not None:
            self.hub.stop()

    #Constructor ==============================================================
    def __init__(self, addr='localhost', hub=True):
        # Before we start, let's kill off any zombies

        if hub:
            # sampy seems to fall over sometimes if 'localhost' isn't specified,
            # even though it shouldn't
            self.hub = sampy.SAMPHubServer(addr=addr)
            self.hub.start()
        else:
            self.hub = None

        self.metadata = {
            'samp.name': 'Python Session',
            'samp.icon.url':'http://docs.scipy.org/doc/_static/scipyshiny_small.png',
            'samp.description.text': 'Python Samp Module',
            'client.version': '0.1a'
        }

        self.client = sampy.SAMPIntegratedClient(metadata=self.metadata, addr=addr)
        self.client.connect()
        atexit.register(self.client.disconnect)

        # Bind interaction functions - we will register that we want to listen
        # for table.highlight.row (the SAMP highlight protocol), and all the
        # typical SAMP-stuff We could include this to listen for other sorts of
        # subscriptions like pointAt(), etc.
        self.client.bind_receive_notification('table.highlight.row', self.highlightRow)
        self.client.bind_receive_notification('table.select.rowList', self.highlightRow)
        self.client.bind_receive_notification('table.load.fits', self.receiveNotification)
        self.client.bind_receive_notification('table.load.csv', self.receiveNotification)
        self.client.bind_receive_notification('samp.app.*', self.receiveNotification)

        self.client.bind_receive_call('samp.app.*', self.receiveCall)
        self.client.bind_receive_call('table.load.fits', self.receiveCall)
        self.client.bind_receive_call('table.load.csv', self.receiveCall)

        self.client.bind_receive_notification('image.load.fits', self.receiveCall)
        self.client.bind_receive_call('image.load.fits', self.receiveCall)

        self.tables = {}
        self.images = {}
        self.lastMessage = None

    # Generic response protocols ===============================================
    def receiveNotification(self, privateKey, senderID, mType, params, extra):
        self.lastMessage = {'label':'Notification',
                            'privateKey':privateKey,
                            'senderID': senderID,
                            'mType': mType,
                            'params': params,
                            'extra': extra }

        print('[SAMP] Notification {0}'.format(self.lastmessage))

        if mType == 'image.load.fits':
            self.receiveCall(privateKey, senderID, None, mType, params, extra)

    def receiveResponse(self, privateKey, senderID, msgID, response):
        print('[SAMP] Response ', privateKey, senderID, msgID, response)

    def receiveCall(self, privateKey, senderID, msgID, mType, params, extra):
        data = {'privateKey':privateKey,
                'senderID':senderID,
                'msgID':msgID,
                'mType':mType,
                'params':params}

        print('[SAMP] Call {0}'.format(data))

        if 'table.load' in mType:
            print("[SAMP] Table received.")
            self.tables[params['name']] = data['params']
            self.tables[params['name']]['data'] = None
        elif 'image.load' in mType:
            print("[SAMP] Image received.")
            self.images[params['name']] = data['params']
            self.images[params['name']]['data'] = None
        else:
                print('[SAMP] Call')
                print(mType)

        self.client.ereply(msgID, sampy.SAMP_STATUS_OK, result={'txt': 'printed' })

    def __getitem__(self, k):
        return self.get(k)

    def __setitem__(self, k, data):
        """ Broadcast data to all """
        return self.send(k, data, to='all')

    def __call__(self):
        """ print detailed info about the current client """
        self.info()
        neighbours = self.getNeighbours()
        if len(neighbours) > 0:
            print("%d detected client(s):" % len(neighbours))
            for n in neighbours:
                print('\t' + self.client.get_metadata(n)['samp.name'])

        print("Registered tables: %d" % len(self.tables))
        for k in self.tables.keys():
            print('      %s' % k)

    # Application functions ===================================================
    def disconnect(self):
        self.client.disconnect()

    def get(self, k):
        """ get table """
        if k in self.tables:
            cTab = self.tables[k]
            if cTab['data'] is None:
                cTab['data'] = data = Table(cTab['url'])
                return data
            else:
                return cTab['data']
        if k in self.images:
            cTab = self.images[k]
            if cTab['data'] is None:
                cTab['data'] = data = Table(cTab['image-id']).data
                return data
            else:
                return cTab['data']

    def send(self, k, data, to='All'):
        """ Broadcast data to one or all applications """

        if k[-5:] != '.fits':
            k += '.fits'

        kwargs = {'clobber': True}

        if Table != SimpleTable:
            kwargs['append'] = False

        if isinstance(data, Table):
            data.write(k, **kwargs)
            tab = data
        else:
            tab = Table(data).write(k, **kwargs)

        self.tables[k] = {'name':k,
                          'url': 'file://' + os.getcwd() + '/' + k,
                          'data': tab
                          }
        return self._broadcastTable(k)

    def info(self):
        """ print information about the current client """
        for k in self.metadata.keys():
            print(k, self.metadata[k])

    def getNeighbours(self):
        """ returns print information about the current client """
        return self.client.get_registered_clients()

    def getAppId(self, name):
        """
        Returns the registered Id of a given application name
        """
        neighbours = self.client.get_registered_clients()
        for neighbour in neighbours:
            metadata = self.client.get_metadata(neighbour)
            try:
                if (metadata['samp.name'] == name):
                    return neighbour
            except KeyError:
                continue

    def isAppRunning(self, name):
        """
        Check if a given application is running and registered to the SAMP
        server
        """
        neighbours = self.client.get_registered_clients()

        for neighbour in neighbours:
            metadata = self.client.get_metadata(neighbour)

            try:
                if (metadata['samp.name'] == name):
                    self.topcat = neighbour
                    return True

            except KeyError:
                continue

        return False

    # Broadcast a table file to TOPCAT
    def _broadcastTable(self, table, to='All'):
        """ Broadcast message to given recipient """
        metadata = {'samp.mtype': 'table.load.fits',
                    'samp.params': {'name': table,
                                    'table-id': table,
                                    'url': 'file://' + os.getcwd() + '/' + table }
                    }

        if len(self.client.get_registered_clients()) > 0:
            if to.lower() == 'all':
                return self.client.notify_all(metadata)
            else:
                if self.isAppRunning(to):
                    return self.client.notify(self.getAppId(to), metadata)
                else:
                    return False

    def highlightRow(self, privateKey, senderID, mType, params, extra):
        """ Not working yet... Only receiving the selected row """
        print('[SAMP] Highlighted row', privateKey, senderID, mType, params, extra)

        self.lastMessage = {'label':'Highlighted Row',
                            'privateKey':privateKey,
                            'senderID': senderID,
                            'mType': mType,
                            'params': params,
                            'extra': extra }

        if(senderID == self.topcat):
            try:
                filename, row = [params['url'], int(params['row'])]
            except KeyError:
                print('[SAMP] Highlighted row was missing vital information')
            else:
                print('TOPCAT tells us that row %s of file %s was highlighted!' % (row, filename))


#==========================================================================
#==========================================================================
def demo(client=None):
    """
    This function is a short example of how to use this package

    It creates a minimal table.
    However it is prefered to use the mytables module to do so, in order to
    provide more data descriptions such as units or comments.
    """
    import numpy
    # first start a topcat session. (Topcat application can be launched after)
    if client is None:
        c  = Client()
    else:
        c = client

    # Some data is generated from my program. Let's use some mock data.
    x = numpy.arange(0, 100)
    y = x ** 2

    # broadcast a table to topcat
    c['t0'] = {'x':x, 'y':y }

    print("""
       to obtain a table from topcat, broadcast a message from topcat and then
       mytable = topcat['mytable']
       """)
    if client is None:
        return c
