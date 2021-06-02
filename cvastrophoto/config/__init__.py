
def get_default_store():
    from .store import files
    from os.path import expanduser
    return files.FilesStore(expanduser('~/.cvastrophoto/config'))


def get_default_profile(store=None):
    return get_profile('default', store)


def get_profile(name, store=None):
    from .profiles import base
    if store is None:
        store = get_default_store()
    return base.RootProfile(store.get_section('profiles').get_section(name), name)
