"""Miscellaneous utility functions (i.e. stuff not specific to this project,
and which doesn't fit in anywhere else)."""

from datetime import datetime
from random import randint
from struct import pack


def unique_key():
    """Return a base64-encoded key of ~10 characters; attempts to be unique."""
    time_bits = 48
    random_bits = 16
    # Everything needs to be packed into 64 bits
    assert time_bits + random_bits == 64
    # Arbitrary choice :)
    epoch = datetime(2015, 8, 21, 15, 15, 47, 123416)

    delta = datetime.utcnow() - epoch
    microseconds = int(delta.total_seconds() * 10**6)
    masked_time = microseconds & ~(1 << time_bits)
    assert masked_time == microseconds, "Increase the epoch or use more bits!"

    random_val = randint(0, (1 << random_bits) - 1)

    key = (masked_time << random_bits) | random_val
    # Make sure thing actually DO go into 64 bits!
    assert key < 1 << 63

    return pack('>Q', key).encode('base64')[:-1]
