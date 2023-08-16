import numpy as np
from itertools import zip_longest


def generate_initial_angles(
    tilt_angle_zero: float = 0,
    tilt_angle_min: float = -90,
    tilt_angle_max: float = 90,
    tilt_angle_step: float = None,
    num_tilt_angles: int = None,
) -> np.ndarray:
    """
    Generate the initial scan angles for a continuous scan

    Args:
        tilt_angle_zero: The zero tilt offset (degrees)
        tilt_angle_max: The maximum tilt angle (degrees)
        tilt_angle_min: The minimum tilt angle (degrees)
        tilt_angle_step: The tilt angle step (degrees)
        num_tilt_angles: The number of tilt angles

    Returns:
        The tilt angles

    """
    # Check the input
    assert tilt_angle_min <= tilt_angle_zero
    assert tilt_angle_zero <= tilt_angle_max
    assert tilt_angle_min >= -90
    assert tilt_angle_max <= 90

    # Generate scan angles symmetrically around the zero tilt angle up the the
    # min and max with the given setp
    if tilt_angle_step is None and num_tilt_angles is None:
        raise RuntimeError("Either tilt_angle_step or num_tilt_angles needs to be set")
    elif tilt_angle_step is not None and num_tilt_angles is not None:
        raise RuntimeError("Cannot set both tilt_angle_step and num_tilt_angles")
    elif tilt_angle_step is not None:
        tilt_angle_min = tilt_angle_zero - tilt_angle_step * np.floor(
            (tilt_angle_zero - tilt_angle_min) / tilt_angle_step
        )
        tilt_angle_max = tilt_angle_zero + tilt_angle_step * np.ceil(
            (tilt_angle_max - tilt_angle_zero) / tilt_angle_step
        )
    elif num_tilt_angles is not None:
        tilt_angle_step = (tilt_angle_max - tilt_angle_min) / num_tilt_angles
    else:
        raise RuntimeError("Programmer Error")

    # Generate the angles
    angles = np.arange(tilt_angle_min, tilt_angle_max, tilt_angle_step)
    assert angles.min() >= -90
    assert angles.max() < 90

    # Return the angles
    return angles


def shuffle_array(x: np.ndarray, n: int) -> np.ndarray:
    """
    Shuffle the array in batches.

    First the array is split into two equally sized arrays. If the input array
    has odd length then the first sub array will be 1 element longer. The
    second sub array is tehn reversed and each of the two sub arrays is further
    split into a number of sub arrays each with n elements. Finally, the
    batches of n elements from each sub array are interleaved and the final
    array is flattened.

    Args:
        x: The array to shuffle
        n: The number in a batch

    Returns:
        The shuffled array

    """

    # Check the input
    assert n > 0
    assert n <= len(x) // 2

    # Split the angles into two lists of equal size
    a, b = np.array_split(x, 2)

    # Split each list into chuncks of size n (whilst reversing the second list)
    a = np.array_split(a, np.arange(n, len(a), n))
    b = np.array_split(np.flip(b), np.arange(n, len(b), n))

    # Interleave the chunks of each list and flatten
    c = np.concatenate(
        [cc for aa, bb in zip_longest(a, b) for cc in [aa, bb] if cc is not None]
    )
    assert len(c) == len(x)

    # Return the shuffled array
    return c


def generate_dose_symmetric_scan(
    angles: np.ndarray, symmetry: int = 0, tilt_angle_zero: float = 0
) -> np.ndarray:
    """
    Reorder the tilt angles for the desired symmetry

    For symmetry == 0, the continous array is returned.

    For symmetry == 1, the standard dose symmetric order is returned.

    For symmetry == n, the dose symmetric array is shuffled n-1 times with
    batches of size 2^i

    Args:
        angles: The array of continious tilt angles
        symmetry: The dose symmetric order
        tilt_angle_zero: The zero tilt angle

    Returns:
        The tilt angles

    """

    # If symmetry is zero then do nothing
    if symmetry > 0:
        # Order angles around zero
        angles = np.array(sorted(angles, key=lambda x: abs(x)))

        # Shuffle the array n-1 times
        for i in range(symmetry - 1):
            angles = shuffle_array(angles, 2**i)

        # Recentre angles around the zero angle
        angles += tilt_angle_zero
        angles = (angles + 90) % 180 - 90

    # Return the angles
    return angles


def generate_spiral_scan(angles: np.ndarray, n: int = 0) -> np.ndarray:
    """
    Generate a spiral scan

    Args:
        angles: The array of continious tilt angles
        n: The number of spirals

    Returns:
        The tilt angles

    """

    # Only do anything if n > 1
    if n > 1:
        # Ensure angles are in sorted order
        angles = np.array(sorted(angles))

        # Do n scans through the angles
        angles = np.concatenate([angles[i::n] for i in range(n)])

    # Return the angles
    return angles


def generate_swinging_scan(angles: np.ndarray, n: int = 0) -> np.ndarray:
    """
    Generate a swinging scan

    Args:
        angles: The array of continious tilt angles
        n: The number of swingings

    Returns:
        The tilt angles

    """

    # Flip odd series
    def flip_or_not(x, i):
        if i % 2 == 0:
            return x[i // 2]
        else:
            return np.flip(x[len(x) - i // 2 - 1])

    # Only do anything if n > 1
    if n > 1:
        # Ensure angles are in sorted order
        angles = np.array(sorted(angles))

        # Split the angles into n interleaved series
        splits = [angles[i::n] for i in range(n)]

        # Order the angles to swinging back and forward
        angles = np.concatenate([flip_or_not(splits, i) for i in range(n)])

    # Return the angles
    return angles


def generate_scan(
    tilt_angle_zero: float = 0,
    tilt_angle_min: float = -90,
    tilt_angle_max: float = 90,
    tilt_angle_step: float = None,
    num_tilt_angles: int = None,
    mode: str = "symmetric",
    symmetry: int = 0,
    skipnum: int = 0,
) -> np.ndarray:
    """
    Generate the scan angles

    Args:
        tilt_angle_zero: The zero tilt offset (degrees)
        tilt_angle_max: The maximum tilt angle (degrees)
        tilt_angle_min: The minimum tilt angle (degrees)
        tilt_angle_step: The tilt angle step (degrees)
        num_tilt_angles: The number of tilt angles
        symmetry: The dose symmetric order
        skipnum: The number of images to skip

    Returns:
        The tilt angles

    """

    def get_skipnum(N, symmetry, skipnum):
        if skipnum == 0:
            if symmetry > 0:
                skipnum = N // (2 ** (symmetry - 1))
        return skipnum

    # Check the input
    assert symmetry >= 0
    assert mode in ["symmetric", "spiral", "swinging"]

    # Generate the set of initial tilt angles
    angles = generate_initial_angles(
        tilt_angle_zero,
        tilt_angle_min,
        tilt_angle_max,
        tilt_angle_step,
        num_tilt_angles,
    )

    # Get the number to skip
    skipnum = get_skipnum(len(angles), symmetry, skipnum)

    # Reorder the angles into a dose symmetric scan
    if mode == "symmetric":
        angles = generate_dose_symmetric_scan(angles, symmetry, tilt_angle_zero)
    elif mode == "spiral":
        angles = generate_spiral_scan(angles, skipnum)
    elif mode == "swinging":
        angles = generate_swinging_scan(angles, skipnum)
    else:
        raise RuntimeError("Programmer Error")

    # Return the angles
    return angles
