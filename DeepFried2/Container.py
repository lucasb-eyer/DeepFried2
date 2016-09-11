import DeepFried2 as df
from collections import OrderedDict as _OrderedDict
from itertools import chain as _chain


class Container(df.Module):

    def __init__(self, *modules):
        df.Module.__init__(self)

        self.modules = []
        if len(modules):
            self.add(*modules)

    def evaluate(self):
        df.Module.evaluate(self)
        for module in self.modules:
            module.evaluate()

    def training(self):
        df.Module.training(self)
        for module in self.modules:
            module.training()

    def parameters(self, *a, **kw):
        params = list(_chain.from_iterable(m.parameters(*a, **kw) for m in self.modules))

        # We actually need to remove duplicates from the list of parameters
        # (and their corresponding gradients) in order to support reusing
        # the same layer at multiple places in the graph,
        # e.g. do weight sharing.
        return list(_OrderedDict.fromkeys(params).keys())

    def get_extra_outputs(self):
        return list(_chain.from_iterable(m.get_extra_outputs() for m in self.modules))

    def get_extra_updates(self):
        return list(_chain.from_iterable(m.get_extra_updates() for m in self.modules))

    def get_stat_updates(self):
        return list(_chain.from_iterable(m.get_stat_updates() for m in self.modules))

    def add(self, *modules):
        for m in modules:
            assert isinstance(m, df.Module), "`{}`s can only contain objects subtyping `df.Module`. You tried to add the following `{}`: {}".format(df.utils.typename(self), df.utils.typename(m), m)
        self.modules += modules

        # Just return for enabling some nicer usage-patterns.
        return modules

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(*self.modules[key])
        elif isinstance(key, (list, tuple)):
            return type(self)(*[self.modules[k] for k in key])
        else:
            return self.modules[key]

    def __len__(self):
        # This one is needed to make __getindex__ work with negative indices.
        return len(self.modules)

    def __getstate__(self):
        return [m.__getstate__() for m in self.modules]

    def __setstate__(self, state):
        for m, s in zip(self.modules, state):
            m.__setstate__(s)


class SingleModuleContainer(Container):
    def __init__(self, module):
        Container.__init__(self, module)

    def add(self, mod):
        if len(self.modules):
            raise TypeError("Container `{}` can't hold more than one module.".format(df.utils.typename(self)))
        Container.add(self, mod)

    def symb_forward(self, symb_input):
        return self.modules[0](symb_input)
