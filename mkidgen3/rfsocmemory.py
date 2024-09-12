import psutil
import blosc2
from mkidgen3.system_parameters import PL_TOTAL_BYTES, SYSTEM_OVERHEAD_BYTES, COMPRESSION_OVERHEAD_BYTES


def memfree_mib():
    return psutil.virtual_memory().available//1024**2


def determine_max_chunk(domain: str, assume_compression=False, demands: list | tuple | None = None) -> int:
    """
    Attempt to estimate how big of a chunk of memory can be allocated w/o swapping.

    Args:
        domain: Are we working in the PS or the PL?
        demands: An iterable of numbers containing existing pressures on the memory
        assume_compression (False) include enough memory for compression of the data in the ps side

    Returns: a number of

    """
    domain = domain.lower()
    if domain not in ('ps', 'pl'):
        raise ValueError('Valid domains are ps and pl')

    total = psutil.virtual_memory().total if domain == 'ps' else PL_TOTAL_BYTES

    if domain == 'ps':
        total -= SYSTEM_OVERHEAD_BYTES

    demands = list(demands) if demands else []

    if assume_compression:
        demands.append(blosc2.MAX_OVERHEAD)

    left = int(total - sum(demands))

    if assume_compression:
        left = min(blosc2.MAX_BUFFERSIZE, left)

    return left
