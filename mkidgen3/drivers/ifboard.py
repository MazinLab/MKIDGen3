import warnings

# We can use UserWarning here instead to defeat the default filter but that is less informative
warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    "Import the ifboard from mkidgen3.equipment_drivers",
    DeprecationWarning,
    stacklevel=0,
)

from mkidgen3.equipment_drivers.ifboard import *
