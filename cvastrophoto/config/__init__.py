from six import string_types as basestring

from os.path import expanduser

from .store import files
from .profiles import base


def get_default_store():
    return files.FilesStore(expanduser('~/.cvastrophoto/config'))


def get_default_profile(store=None):
    return get_profile('default', store)


def get_profile(name, store=None):
    if store is None:
        store = get_default_store()
    return base.RootProfile(store.get_section('profiles').get_section(name), name)


def profile_from(name_or_profile=None, **kw):
    """ Construct a profile

    Returns a profile based on a configuration-supplied value. The behavior will
    be polymorphic on the kind of value provided.

    If None, the default profile is returned.

    If a string, a profile of that name is looked up.

    If an instance, it is just returned as is.
    """
    if name_or_profile is None:
        return get_default_profile(**kw)
    elif isinstance(name_or_profile, basestring):
        return get_profile(name_or_profile, **kw)
    elif issubclass(base.BaseProfile):
        return name_or_profile
    else:
        raise ValueError(
            "profile_from can take a string or profile instance, not %r" % (type(name_or_profile).__name__,)
        )


def name_from(device):
    if device is None:
        return None
    elif isinstance(device, basestring):
        return device
    else:
        return getattr(device, 'name', None)
