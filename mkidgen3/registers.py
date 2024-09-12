import enum

from typing import Type, Optional


def getmask(width: int):
    """Get an int with the low `width` bits set

    Parameters
    ----------
    width : int
        The number of bits to set in the output
    """
    return (1 << width) - 1


def checkslice(sl: slice, width: int):
    """Checks that a slice is valid for a particular register width

    Parameters
    ----------
    sl : slice
        The slice to check
    width : int
        The underlying register width
    """
    if not (sl.step is None) and sl.stride != 1:
        raise IndexError("Access Stride must be 1 or None")
    if sl.stop > width or sl.start >= width:
        raise IndexError("Access wider than register width ({:d}".format(width))
    if sl.stop < sl.start:
        raise IndexError("Reverse indexing unsupported")
    if sl.stop < 0 or sl.start < 0:
        raise IndexError("Negative indicies unsupported")


class MetaRegister:
    """A metaclass for both `Register` and `Field`"""

    _objcache = None

    def __get__(self, obj, objtype=None):
        pass

    def __set__(self, obj, val):
        pass

    def __len__(self):
        pass

    def __getitem__(self, obj, sl: int | slice):
        """Extract a field from a register

        Get a slice of this register, this is only callable from subclasses
        of this class, if you want to extract a field on the fly you can do
        that with:

        ```python
        reg = Register(...)
        field_data = field(slice(0, 4), reg).__get__(self)
        ```

        Where `self` is a reference to the IP Object

        Parameters
        ----------
        obj
            A reference to the underlying IP
        sl : int | slice
            The bit(s) to access

        Returns
        -------
        int : the requested field
        """
        self._objcache = obj
        if type(sl) is int:
            val = self.__get__(obj)
            return (val >> sl) & 1
        checkslice(sl, self.__len__())
        if sl.start is None:
            return self.__getitem__(obj, self.slice(0, sl.stop))
        if sl.stop == sl.start:
            return self.__getitem__(obj, sl.start)
        mask = getmask(sl.stop - sl.start)
        val = self.__get__(obj)
        return mask & (val >> sl.start)

    def __setitem__(self, obj, sl: int | slice, val: int):
        """Sets a field to a given value, see `MetaRegister.__get__`

        Parameters
        ----------
        obj
            A reference to the underlying IP
        sl : int | slice

        """
        self._objcache = obj
        if type(sl) is int:
            val_init = self.__get__(obj)
            self.__set__(obj, (val_init & (getmask(32) ^ (1 < sl))) | (val << sl))
            return
        checkslice(sl, self.__len__())
        if sl.start is None:
            return self.__setitem__(obj, self.slice(0, sl.stop), val)
        if sl.stop == sl.start:
            return self.__setitem__(obj, self.slice(0, sl.stop), val)
        mask = getmask(sl.stop - sl.start)
        val_init = self.__get__(obj)
        self.__set__(
            obj,
            (val_init & (getmask(32) ^ (mask << sl.start))) | val << sl.start,
        )


class Register(MetaRegister):
    """A read write register referencing an underlying PYNQ register

    This exists to make building pynq IP marginally more ergonomic, it allows
    easy creation of fields at runtime that don't exist in an HWH description

    When accessing the underlying register in an IP it should always look like
    an int, accessing always reads and assignments always write, methods other
    than `__get__` and `__set__` are typically only available to fields of this
    register

    Typically instead of instantiating this directly you will want to use the
    `register` decorator on a method in your class, if you have an underlying
    register in the HWH description for an IP named `my_reg` you would
    instantiate this like:

    ```python
    import registers

    class MyIP(pynq.DefaultIP):
        ...
        @registers.register
        def my_reg(self):
            return self.register_map.my_reg
    ```

    This slightly odd API exists because the registers can't be resolved
    until the class is instantiated and the HWH is parsed
    """

    def __init__(self, getreg):
        """
        Initilize the class with a function that returns the underlying pynq
        register, this function will recieve one argument, a reference to the
        underlying IP which this register is an attribute of

        If you would like to instantiate this with a register not contained
        in the HWH file you can do at some offeset `OFFSET`:
        ```python
        my_custom_reg = registers.Register(lambda self: pynq.Register(OFFSET, buffer=self.mmio.array))
        ```

        Which would be equivalent to:
        ```python
        class MyIP(DefaultIP):
            ...
            @registers.register
            def my_custom_reg(self):
                return pynq.Register(OFFSET, buffer=self.mmio.array)
        ```

        Arguments
        ---------
        getreg : func(self)
            A function which is passed an argument to the enclosing IP and returns
            a pynq register
        """
        self.getreg = getreg

    def __len__(self):
        """Get the width of this register in bits"""
        # Return a sensible default, maybe should error
        if self._objcache is None:
            return 32
        return self.getreg(self._objcache).width

    def __get__(self, obj, objtype=None):
        """Read the current value of this register"""
        self._objcache = obj
        if obj is None or ((objtype is not None) and issubclass(objtype, MetaRegister)):
            return self
        return self.getreg(obj)[:]

    def __set__(self, obj, val: int):
        """Set the Value of this register"""
        self._objcache = obj
        self.getreg(obj)[:] = val


class RegisterRO(Register):
    """A read only register, can be created with the decorator `register_ro`"""

    def __set__(self, obj, val: int):
        raise ValueError("Attempted to write read only register")


class RegisterWO(Register):
    """A write only register, can be created with the decorator `register_wo`"""

    def __get__(self, obj, objtype=None):
        raise ValueError("Attempted to read write only register")


class RegisterShadow(Register):
    """A write only register with a shadow backing store

    Writes write to the underlying register, reads read the last value written
    and error if there haven't been writes since instantiation

    This can be created with decorator `register_shadow` which can also be used
    to instantiate a version of this class with a default value
    """

    _shadow = None

    def __set__(self, obj, val: int):
        self._shadow = val
        super().__set__(obj, val)

    def __get__(self, obj, objtype=None):
        if self._shadow is None:
            raise ValueError("Shadow register not initilized and never written to")
        return self._shadow


def register(getreg):
    """A decorator which can be used to create a read write register

    For usage see the documentation for `Register`

    Parameters
    ----------
    getreg : func(self)
        A function which takes a reference to the enclosing IP and returns a pynq register
    """
    return Register(getreg)


def register_ro(getreg):
    """See `register`"""
    return RegisterRO(getreg)


def register_wo(getreg):
    """See `register`"""
    return RegisterWO(getreg)


def register_shadow(arg):
    """A decorator which creates an optionally initilized shadow register

    When used like the other decorators this produces an uninit shadow register.

    To create a shadow register initilized to some value (in this case 128):
    ```python
    class MyIP(DefaultIP):
        ...
        @registers.register_shadow(128)
        def my_shadow_reg(self):
            return self.registers.my_write_only_register
    ```
    """
    if type(arg) is int:

        class RegisterShadowInit(RegisterShadow):
            _shadow = arg

        return lambda getreg: RegisterShadowInit(getreg)
    return RegisterShadow(arg)


class Field(MetaRegister):
    """A class representing an individual register field

    The backing parent can be a mkidgen3 `Register` or any other concerete
    subclass of `MetaRegister`. The recommended way to create this is to
    use the `field` decorator.

    There are two ways to approach this, the recommended way exists for
    consistency with the `Register` interface:

    ```python
    import mkidgen3.registers as registers

    class MyIP(DefaultIP):
        ...
        @registers.register_ro
        def status_register(self):
            return self.register_map.status_register
        ...
        # Interrupt Count is bits [0-7] inclusive of the status register
        @registers.field(slice(0, 8))
        def interrupt_count(self):
            return self.status_register
        ...
    ```

    Because mkidgen3 style registers will typically be definition-time
    resolvable, the alternate invocation style is:

    ```python
    class MyIP(DefaultIP):
        ...
        interrupt_count = field(slice(0, 8), status_register)
        ...
    ```
    """

    def __init__(self, parentfunc, sl: slice | int):
        self.parentfunc = parentfunc
        self.sl = sl

    def __get__(self, obj, objtype=None):
        self._objcache = obj
        return self.parentfunc(obj).__getitem__(obj, self.sl)

    def __set__(self, obj, val):
        self._objcache = obj
        self.parentfunc(obj).__setitem__(obj, self.sl, val)


class FieldEnum(Field):
    """This is a field representing an one of several enumerated options

    This is handy in many cases when a register represents one of several
    options, for example:

    ```
    import mkidgen3.registers as registers
    from enum import Enum

    class CaptureSource(Enum):
        NONE = 0
        PHASE = 1
        IQ = 2
        RAW_PHASE = 3

    class MyIP(DefaultIP):
        ...
        @registers.register_ro
        def status_register(self):
            return self.register_map.status_register
        ...
        # The captue source is bits [8:10] inclusive of the status register
        capture_source = field_enum(slice(8, 11), CaptureSource, status_Register)
        ...
    ```
    """

    def __init__(self, parentfunc, sl: slice | int, enum: Type[enum.Enum]):
        super().__init__(parentfunc, sl)
        self.enum = enum

    def __get__(self, obj, objtype=None):
        self._objcache = obj
        return self.enum(super().__get__(obj, objtype))

    def __set__(self, obj, val):
        self._objcache = obj
        if not type(val) is self.enum:
            raise TypeError(
                "Value passed must be of type {:s}, got {:s} ({:s})".format(
                    repr(self.enum), repr(val), repr(type(val))
                )
            )
        super().__set__(obj, val.value)


class FieldBool(Field):
    """This is a single bit field representing a boolean

    To instantiate this you typically want the `field_bool` decorator/factory func
    ```
    import mkidgen3.registers as registers
    ...
    class MyIP(DefaultIP):
        ...
        @registers.register_ro
        def status_register(self):
            return self.register_map.status_register
        ...
        # Bit 11 of the status register indicates if the active capture has completed
        capture_complete = field_bool(11, status_register)
        ...
    ```
    """

    def __init__(self, parentfunc, sl: int):
        super().__init__(parentfunc, sl)

    def __get__(self, obj, objtype=None):
        return bool(super().__get__(obj, objtype))

    def __set__(self, obj, val):
        super().__set__(obj, int(val))


def field(bits: slice | int, reg: Optional[Type[MetaRegister]] = None):
    """Decorator for creating a field from a register and a slice, see `Field`"""
    if reg is not None:
        return Field(lambda _: reg, bits)
    return lambda regfunc: Field(regfunc, bits)


def field_enum(bits: slice | int, enum, reg: Optional[Type[MetaRegister]] = None):
    """Decorator for creating a field from a register, slice, and enum, see `FieldEnum`"""
    if reg is not None:
        return FieldEnum(lambda _: reg, bits, enum)
    return lambda regfunc: FieldEnum(regfunc, bits, enum)


def field_bool(bit: int, reg: Optional[Type[MetaRegister]] = None):
    """Decorator for creating a boolean field from a register and bit index, see `FieldBool`"""
    if reg is not None:
        return FieldBool(lambda _: reg, bit)
    return lambda regfunc: FieldBool(regfunc, bit)
