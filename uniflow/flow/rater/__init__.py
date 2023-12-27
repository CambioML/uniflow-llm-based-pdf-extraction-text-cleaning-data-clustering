"""Rater __init__ Module."""

# this register all possible model server into ModelServerFactory through
# ModelServerFactory.register(cls.__name__, cls) in AbsModelServer
# __init_subclass__


from uniflow.flow.rater.rater_flow import RaterFlow

__all__ = [
    "RaterFlow",
]
