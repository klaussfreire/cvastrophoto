from six import iteritems


class BaseCatalog(object):

    CONCRETE = False
    PRIORITY = 0

    # Offline catalogs are those that don't require external services
    OFFLINE = False

    # Defaultable catalogs are those that can be constructed without arguments
    DEFAULTABLE = False

    def get_object(self, name):
        """ Get a single object given its name

        This method should return an unambiguous object given a name for it.
        It should return a similar result search_name would return given
        the name as a query, if search_name would return a single object.

        If the result is ambiguous it should not return anything, except
        if there's an exact match by name.

        Ie: get_object('M8') would return the object M8, while search_name('M8')
        would yield that one, plus M87 and others that might match given
        the search terms. In this case get_object can return M8 because it's an
        exact match, but if there wasn't one it must not.
        """
        single_match = True
        rv = None
        for obj in self.search_name(name):
            if name in obj.names:
                return obj
            elif rv is None:
                rv = obj
            else:
                single_match = False
        if single_match:
            return rv

    def search_name(self, terms):
        """ Search objects given a search term

        Search object matching the given search term. The search term interpretation
        is implementation dependent for the catalogs, but it must always be a string
        and should accept simple object name searches, either by catalog number
        or common names. Fuzzy matches are allowed.

        Returns a generator over search results, ordered by relevance, from most
        relevant to least relevant.
        """
        raise NotImplementedError()
        yield

    def search_nearby(self, coords, radius, **kw):
        """ Search objects given a coordinate and angular search distance

        Returns a generator over search results, in no particular order,
        of objects inside the cone defined by coords and radius.

        The search may accept additional filter keywords.
        When given, the filters will try to match an arbitrary object's attribute
        to the given filter (say, object_type).

        Filter values may be sets or singular strings. In the case of sets,
        contains will be used to check, and in the case of strings a simple
        comparison against the attribute's value.
        """
        raise NotImplementedError()
        yield

    def _eval_filter(self, obj, **kw):
        set_types = (set, frozenset)
        for attr, query_val in iteritems(kw):
            obj_val = getattr(obj, attr, None)
            if query_val is None != obj_val is None:
                return False
            if query_val is not None and obj_val is not None:
                if not isinstance(query_val, set_types):
                    query_val = (query_val,)
                if obj_val not in query_val:
                    return False
        else:
            return True

    @classmethod
    def concrete_subclasses(cls):
        for scls in cls.__subclasses__():
            if scls.CONCRETE:
                yield scls
            for sscls in scls.concrete_subclasses():
                yield sscls


class CatalogObject(object):

    def __init__(self, name, alt_names=None, descriptive_name=None, **kw):
        self.alt_names = alt_names
        self.name = name
        self.names = names = {name}
        self.descriptive_name = descriptive_name or name
        if alt_names:
            names.update(alt_names)
        if descriptive_name:
            names.add(descriptive_name)
        self.__dict__.update(kw)

    def __str__(self):
        return self.descriptive_name

    def __repr__(self):
        return 'CatalogObject(%r)' % (self.name,)


class MultiCatalog(BaseCatalog):

    def __init__(self, catalogs):
        self.catalogs = catalogs

    def get_object(self, name, **kw):
        for cat in self.catalogs:
            rv = cat.get_object(name, **kw)
            if rv is not None:
                return rv

    def search_name(self, terms, **kw):
        for cat in self.catalogs:
            for obj in cat.search_name(terms, **kw):
                yield obj

    def search_nearby(self, coords, radius, **kw):
        for cat in self.catalogs:
            for obj in cat.search_nearby(coords, radius, **kw):
                yield obj

    @classmethod
    def all(cls, offline=False):
        cat_classes = [
            scls for scls in BaseCatalog.concrete_subclasses()
            if (not offline or scls.OFFLINE) and scls.DEFAULTABLE
        ]
        cat_classes.sort(key=lambda cat:-cat.PRIORITY)
        catalogs = [scls() for scls in cat_classes]
        return cls(catalogs)
