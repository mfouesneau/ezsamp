# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# TODO: this file should be refactored to use a more thread-safe and
# race-condition-safe lockfile mechanism.

import datetime
import os
import sys
import socket
import stat
import warnings
from contextlib import contextmanager

from .six.moves.urllib.parse import urlparse
from .six.moves import xmlrpc_client as xmlrpc
from .six import PY2, PY3
from . import six

from .logger import get_default_logger
log = get_default_logger()

from .constants import SSL_SUPPORT
from .errors import SAMPHubError, SAMPWarning

if SSL_SUPPORT:
    import ssl
    SSL_EXCEPTIONS = (ssl.SSLError,)
else:
    SSL_EXCEPTIONS = ()


@contextmanager
def ignored(*exceptions):
    """A context manager for ignoring exceptions.  Equivalent to::

        try:
            <body>
        except exceptions:
            pass

    Example::
        >>> with ignored(OSError):
        ...     os.remove('file-that-does-not-exist')
    """

    try:
        yield
    except exceptions:
        pass


def _find_home():
    """ Locates and return the home directory (or best approximation) on this
    system.

    Raises
    ------
    OSError
        If the home directory cannot be located - usually means you are running
        Astropy on some obscure platform that doesn't have standard home
        directories.
    """

    # this is used below to make fix up encoding issues that sometimes crop up
    # in py2.x but not in py3.x
    if PY2:
        decodepath = lambda pth: pth.decode(sys.getfilesystemencoding())
    elif PY3:
        decodepath = lambda pth: pth

    # First find the home directory - this is inspired by the scheme ipython
    # uses to identify "home"
    if os.name == 'posix':
        # Linux, Unix, AIX, OS X
        if 'HOME' in os.environ:
            homedir = decodepath(os.environ['HOME'])
        else:
            raise OSError('Could not find unix home directory to search for '
                          'astropy config dir')
    elif os.name == 'nt':  # This is for all modern Windows (NT or after)
        if 'MSYSTEM' in os.environ and os.environ.get('HOME'):
            # Likely using an msys shell; use whatever it is using for its
            # $HOME directory
            homedir = decodepath(os.environ['HOME'])
        # Next try for a network home
        elif 'HOMESHARE' in os.environ:
            homedir = decodepath(os.environ['HOMESHARE'])
        # See if there's a local home
        elif 'HOMEDRIVE' in os.environ and 'HOMEPATH' in os.environ:
            homedir = os.path.join(os.environ['HOMEDRIVE'],
                                   os.environ['HOMEPATH'])
            homedir = decodepath(homedir)
        # Maybe a user profile?
        elif 'USERPROFILE' in os.environ:
            homedir = decodepath(os.path.join(os.environ['USERPROFILE']))
        else:
            try:
                from .six.moves import winreg as wreg
                shell_folders = r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
                key = wreg.OpenKey(wreg.HKEY_CURRENT_USER, shell_folders)

                homedir = wreg.QueryValueEx(key, 'Personal')[0]
                homedir = decodepath(homedir)
                key.Close()
            except:
                # As a final possible resort, see if HOME is present
                if 'HOME' in os.environ:
                    homedir = decodepath(os.environ['HOME'])
                else:
                    raise OSError('Could not find windows home directory to '
                                  'search for astropy config dir')
    else:
        # for other platforms, try HOME, although it probably isn't there
        if 'HOME' in os.environ:
            homedir = decodepath(os.environ['HOME'])
        else:
            raise OSError('Could not find a home directory to search for '
                          'astropy config dir - are you on an unspported '
                          'platform?')
    return homedir


def read_lockfile(lockfilename):
    """
    Read in the lockfile given by ``lockfilename`` into a dictionary.
    """
    # lockfilename may be a local file or a remote URL, but
    # get_readable_fileobj takes care of this.
    lockfiledict = {}
    with open(lockfilename) as f:
        for line in f:
            if not line.startswith("#"):
                kw, val = line.split("=")
                lockfiledict[kw.strip()] = val.strip()
    return lockfiledict


def write_lockfile(lockfilename, lockfiledict):

    lockfile = open(lockfilename, "w")
    lockfile.close()
    os.chmod(lockfilename, stat.S_IREAD + stat.S_IWRITE)

    lockfile = open(lockfilename, "w")
    now_iso = datetime.datetime.now().isoformat()
    lockfile.write("# SAMP lockfile written on %s\n" % now_iso)
    lockfile.write("# Standard Profile required keys\n")
    for key, value in six.iteritems(lockfiledict):
        lockfile.write("{0}={1}\n".format(key, value))
    lockfile.close()


def create_lock_file(lockfilename=None, mode=None, hub_id=None,
                     hub_params=None):

    # Remove lock-files of dead hubs
    remove_garbage_lock_files()

    lockfiledir = ""

    # CHECK FOR SAMP_HUB ENVIRONMENT VARIABLE
    if "SAMP_HUB" in os.environ:
        # For the time being I assume just the std profile supported.
        if os.environ["SAMP_HUB"].startswith("std-lockurl:"):

            lockfilename = os.environ["SAMP_HUB"][len("std-lockurl:"):]
            lockfile_parsed = urlparse(lockfilename)

            if lockfile_parsed[0] != 'file':
                warnings.warn("Unable to start a Hub with lockfile %s. "
                              "Start-up process aborted." % lockfilename,
                              SAMPWarning)
                return False
            else:
                lockfilename = lockfile_parsed[2]
    else:

        # If it is a fresh Hub instance
        if lockfilename is None:

            log.debug("Running mode: " + mode)

            if mode == 'single':
                lockfilename = os.path.join(_find_home(), ".samp")
            else:

                lockfiledir = os.path.join(_find_home(), ".samp-1")

                # If missing create .samp-1 directory
                try:
                    os.mkdir(lockfiledir)
                except OSError:
                    pass  # directory already exists
                finally:
                    os.chmod(lockfiledir,
                             stat.S_IREAD + stat.S_IWRITE + stat.S_IEXEC)

                lockfilename = os.path.join(lockfiledir,
                                            "samp-hub-%s" % hub_id)

        else:
            log.debug("Running mode: multiple")

    hub_is_running, lockfiledict = check_running_hub(lockfilename)

    if hub_is_running:
        warnings.warn("Another SAMP Hub is already running. Start-up process "
                      "aborted.", SAMPWarning)
        return False

    log.debug("Lock-file: " + lockfilename)

    write_lockfile(lockfilename, hub_params)

    return lockfilename


def get_main_running_hub():
    """
    Get either the hub given by the environment variable SAMP_HUB, or the one
    given by the lockfile .samp in the user home directory.
    """
    hubs = get_running_hubs()

    if not hubs:
        raise SAMPHubError("Unable to find a running SAMP Hub.")

    # CHECK FOR SAMP_HUB ENVIRONMENT VARIABLE
    if "SAMP_HUB" in os.environ:
        # For the time being I assume just the std profile supported.
        if os.environ["SAMP_HUB"].startswith("std-lockurl:"):
            lockfilename = os.environ["SAMP_HUB"][len("std-lockurl:"):]
        else:
            raise SAMPHubError("SAMP Hub profile not supported.")
    else:
        lockfilename = os.path.join(_find_home(), ".samp")

    return hubs[lockfilename]


def get_running_hubs():
    """
    Return a dictionary containing the lock-file contents of all the currently
    running hubs (single and/or multiple mode).

    The dictionary format is:

    ``{<lock-file>: {<token-name>: <token-string>, ...}, ...}``

    where ``{<lock-file>}`` is the lock-file name, ``{<token-name>}`` and
    ``{<token-string>}`` are the lock-file tokens (name and content).

    Returns
    -------
    running_hubs : dict
        Lock-file contents of all the currently running hubs.
    """

    hubs = {}
    lockfilename = ""

    # HUB SINGLE INSTANCE MODE

    # CHECK FOR SAMP_HUB ENVIRONMENT VARIABLE
    if "SAMP_HUB" in os.environ:
        # For the time being I assume just the std profile supported.
        if os.environ["SAMP_HUB"].startswith("std-lockurl:"):
            lockfilename = os.environ["SAMP_HUB"][len("std-lockurl:"):]
    else:
        lockfilename = os.path.join(_find_home(), ".samp")

    hub_is_running, lockfiledict = check_running_hub(lockfilename)

    if hub_is_running:
        hubs[lockfilename] = lockfiledict

    # HUB MULTIPLE INSTANCE MODE

    lockfiledir = ""

    lockfiledir = os.path.join(_find_home(), ".samp-1")

    if os.path.isdir(lockfiledir):
        for filename in os.listdir(lockfiledir):
            if filename.startswith('samp-hub'):
                lockfilename = os.path.join(lockfiledir, filename)
                hub_is_running, lockfiledict = check_running_hub(lockfilename)
                if hub_is_running:
                    hubs[lockfilename] = lockfiledict

    return hubs


def check_running_hub(lockfilename):
    """
    Test whether a hub identified by ``lockfilename`` is running or not.

    Parameters
    ----------
    lockfilename : str
        Lock-file name (path + file name) of the Hub to be tested.

    Returns
    -------
    is_running : bool
        Whether the hub is running
    hub_params : dict
        If the hub is running this contains the parameters from the lockfile
    """

    is_running = False
    lockfiledict = {}

    # Check whether a lockfile alredy exists
    try:
        lockfiledict = read_lockfile(lockfilename)
    except IOError:
        return is_running, lockfiledict

    if "samp.hub.xmlrpc.url" in lockfiledict:
        try:
            proxy = xmlrpc.ServerProxy(lockfiledict["samp.hub.xmlrpc.url"]
                                       .replace("\\", ""), allow_none=1)
            proxy.samp.hub.ping()
            is_running = True
        except xmlrpc.ProtocolError:
            # There is a protocol error (e.g. for authentication required),
            # but the server is alive
            is_running = True
        except SSL_EXCEPTIONS:
            # SSL connection refused for certifcate reasons...
            # anyway the server is alive
            is_running = True
        except socket.error:
            pass

    return is_running, lockfiledict


def remove_garbage_lock_files():

    lockfilename = ""

    # HUB SINGLE INSTANCE MODE

    lockfilename = os.path.join(_find_home(), ".samp")

    hub_is_running, lockfiledict = check_running_hub(lockfilename)

    if not hub_is_running:
        # If lockfilename belongs to a dead hub, then it is deleted
        if os.path.isfile(lockfilename):
            with ignored(OSError):
                os.remove(lockfilename)

    # HUB MULTIPLE INSTANCE MODE

    lockfiledir = os.path.join(_find_home(), ".samp-1")

    if os.path.isdir(lockfiledir):
        for filename in os.listdir(lockfiledir):
            if filename.startswith('samp-hub'):
                lockfilename = os.path.join(lockfiledir, filename)
                hub_is_running, lockfiledict = check_running_hub(lockfilename)
                if not hub_is_running:
                    # If lockfilename belongs to a dead hub, then it is deleted
                    if os.path.isfile(lockfilename):
                        with ignored(OSError):
                            os.remove(lockfilename)
