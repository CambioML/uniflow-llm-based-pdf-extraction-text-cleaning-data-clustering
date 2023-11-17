"""Model __init__ Module."""

# this register all possible model server into ModelServerFactory through
# ModelServerFactory.register(cls.__name__, cls) in AbsModelServer
# __init_subclass__
from uniflow.model.server import *  # noqa: F401, F403
