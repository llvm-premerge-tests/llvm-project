import inspect


def print_gen_init_arg_names(cls: type):
    """
    Prints the argument names of the __init__ function of the generated base
    class of a mix-in extension.

    This is useful for ensuring that mix-in classes remain in sync with their
    underlying op. If operands are added or existing ones are renamed in that
    op, the same change needs to occur in the mix-in class. However, in many
    cases, these changes do not otherwise break the tests of the mix-ins, such
    that the mix-in classes become out of sync. This function allows to print
    the argument names of the generated base class, which can then be checked
    for in a `CHECK` statement.

    If you are modifying a tablegen definition and that modification breaks a
    test that uses this function, please update the corresponding mix-in class
    in the `*_ext.py` of the dialect.
    """
    mro = cls.__mro__[2]
    assert mro.__module__.endswith("_gen")
    print(list(inspect.signature(mro.__init__).parameters.keys()))
