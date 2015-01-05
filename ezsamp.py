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

try:
    from astropy.vo import samp as sampy
except ImportError:
    from . import sampy

from .simpletable import SimpleTable

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
    iteritems = operator.methodcaller('iteritems')
    itervalues = operator.methodcaller('itervalues')

try:
    from eztables import Table
except Exception as e:
    print('Warning: eztable could not be imported for the following reason: {0}'.format(e))
    print('Warning: switching to SimpleTable instead')
    Table = SimpleTable


# ==========================================================================
# HUB -- Generate a SAMP hub to manage communications
# ==========================================================================
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


# ==========================================================================
# Client -- Generate a SAMP client able to send and receive data
# ==========================================================================
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
    # Destructor ===============================================================
    def __del__(self):
        self.client.disconnect()
        if self.hub is not None:
            self.hub.stop()

    # Constructor ==============================================================
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


# ==========================================================================
# ==========================================================================
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
