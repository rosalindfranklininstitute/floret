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
        # Compute the maximum tilt around the zero tilt
        max_tilt = min(
            tilt_angle_zero - tilt_angle_min, tilt_angle_max - tilt_angle_zero
        )

        # Compute the minimum tilt angle
        tilt_angle_min = tilt_angle_zero - tilt_angle_step * np.floor(
            (max_tilt) / tilt_angle_step
        )

        # Compute the maximum tilt angle
        tilt_angle_max = tilt_angle_zero + tilt_angle_step * np.ceil(
            (max_tilt) / tilt_angle_step
        )

    elif num_tilt_angles is not None:
        tilt_angle_step = (tilt_angle_max - tilt_angle_min) / num_tilt_angles
    else:
        raise RuntimeError("Programmer Error")

    # Generate the angles
    assert tilt_angle_min < tilt_angle_max
    assert tilt_angle_step > 0
    angles = np.arange(tilt_angle_min, tilt_angle_max, tilt_angle_step)
    angles = np.sort(angles)
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

    # Ensure angles are in sorted order
    order = np.arange(len(angles))

    # If symmetry is zero then do nothing
    if symmetry > 0:
        # Order angles around zero
        order = np.array(sorted(order, key=lambda x: abs(angles[x] - tilt_angle_zero)))

        # Shuffle the array n-1 times
        for i in range(symmetry - 1):
            order = shuffle_array(order, 2**i)

    # Return the angles
    return order


def generate_spiral_scan(angles: np.ndarray, n: int = 0) -> np.ndarray:
    """
    Generate a spiral scan

    Args:
        angles: The array of continious tilt angles
        n: The number of spirals

    Returns:
        The tilt angles

    """
    # Ensure angles are in sorted order
    order = np.arange(len(angles))

    # Only do anything if n > 1
    if n > 1:
        # Do n scans through the angles
        order = np.concatenate([order[i::n] for i in range(n)])

    # Return the angles
    return order


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

    # Ensure angles are in sorted order
    order = np.arange(len(angles))

    # Only do anything if n > 1
    if n > 1:
        # Split the angles into n interleaved series
        splits = [order[i::n] for i in range(n)]

        # Order the angles to swinging back and forward
        order = np.concatenate([flip_or_not(splits, i) for i in range(n)])

    # Return the angles
    return order


def generate_initial_positions(nhelix: int, nangles: int) -> np.ndarray:
    """
    Generate the initial positions

    Args:
        nhelix: The nhelix parameter defining the sub shifts in position
        nangles: The number of angles to sample

    Returns:
        A position for each angle

    """
    assert nhelix > 0
    assert nangles > 0
    return np.array(
        [
            i / nhelix
            for j in range(0, nangles, nhelix)
            for i in range(min(nangles - j, nhelix))
        ]
    )


def generate_shifted_positions(
    angles: np.ndarray,
    positions: np.ndarray,
    position_min: int = 0,
    position_max: int = 1,
) -> tuple:
    """
    Generate the shifted positions

    Args:
        angles: The list of angles
        positions: The list of positions
        position_min: The minimum normalised position
        position_max: The maximum normalised position

    Returns:
        A tuple with the list of angles and list of positions

    """
    position_diff = position_max - position_min
    assert position_diff > 0
    positions = np.stack([positions + p for p in range(position_min, position_max)])
    angles = np.tile(angles, (positions.shape[0], 1))
    assert angles.shape == positions.shape
    return angles, positions


def generate_final_order(
    angles: np.ndarray,
    positions: np.ndarray,
    order_by: str = "angle",
    interleave_positions: bool = True,
) -> tuple:
    """
    Generate the final order

    If order_by=angle then for each angle, all positions are acquired.
    If order_by=position then for each position, all angles are acquired.

    If order_by=angle and interleave_positions=True then we skip adjacent positions
    and collect angles twice. We do this because the beam is larger than the collection
    area so adjacent positions will overlap.

    Args:
        angles: The list of angles
        positions: The list of positions
        order_by: Order acquisition by position or angle
        interleave_positions: Interleave the positions

    Returns:
        A tuple with the list of angles and list of positions

    """
    # Swap order if desired and then flatten array
    if order_by == "angle":
        positions = positions.T
        angles = angles.T

        # Interleave positions by skipping one position if desired
        if interleave_positions:
            positions = np.concatenate([positions[:, i::2].flatten() for i in range(2)])
            angles = np.concatenate([angles[:, i::2].flatten() for i in range(2)])

    # Flatten the lists
    angles = angles.flatten()
    positions = positions.flatten()

    # Return the final list of angles and positions
    return angles, positions


def generate_scan(
    tilt_angle_zero: float = 0,
    tilt_angle_min: float = -90,
    tilt_angle_max: float = 90,
    tilt_angle_step: float = None,
    num_tilt_angles: int = None,
    mode: str = "symmetric",
    symmetry: int = 0,
    stepnum: int = 1,
    nhelix: int = 1,
    position_min: int = 0,
    position_max: int = 1,
    order_by="angle",
    interleave_positions=True,
) -> tuple:
    """
    Generate the scan angles

    Args:
        tilt_angle_zero: The zero tilt offset (degrees)
        tilt_angle_max: The maximum tilt angle (degrees)
        tilt_angle_min: The minimum tilt angle (degrees)
        tilt_angle_step: The tilt angle step (degrees)
        num_tilt_angles: The number of tilt angles
        symmetry: The dose symmetric order
        stepnum: The number of images to step
        nhelix: The nhelix parameter
        position_min: The minimum normalised position
        position_max: The maximum normalised position
        order_by: Order by angle or position
        interleave_positions: Skip and interleave adjacent positions if order_by=angle

    Returns:
        The tilt angles

    """

    def get_stepnum(N, symmetry, stepnum):
        if stepnum == 0:
            if symmetry > 0:
                stepnum = N // (2 ** (symmetry - 1))
        return stepnum

    # Check the input
    assert symmetry >= 0
    assert mode in ["symmetric", "spiral", "swinging"]
    assert nhelix >= 1
    assert (position_max - position_min) >= 1

    # Generate the set of initial tilt angles
    angles = generate_initial_angles(
        tilt_angle_zero,
        tilt_angle_min,
        tilt_angle_max,
        tilt_angle_step,
        num_tilt_angles,
    )

    # Assign a fractional position to each tilt angle for the n-helix
    positions = generate_initial_positions(nhelix, len(angles))

    # Get the number to step
    stepnum = get_stepnum(len(angles), symmetry, stepnum)

    # Reorder the angles into a dose symmetric scan
    if mode == "symmetric":
        order = generate_dose_symmetric_scan(
            angles, symmetry, float(angles[len(angles) // 2])
        )
    elif mode == "spiral":
        order = generate_spiral_scan(angles, stepnum)
    elif mode == "swinging":
        order = generate_swinging_scan(angles, stepnum)
    else:
        raise RuntimeError("Programmer Error")

    # Order the angles and positions
    angles = angles[order]
    positions = positions[order]

    # Generate the beam shifted positions along the pillar
    angles, positions = generate_shifted_positions(
        angles, positions, position_min, position_max
    )

    # Swap order if desired and then flatten array
    angles, positions = generate_final_order(
        angles, positions, order_by, interleave_positions
    )

    # Return the angles
    return positions, angles
