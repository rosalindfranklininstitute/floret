import floret
import floret.command_line
import numpy as np


def test_generate_initial_angles():
    angles = floret.generate_initial_angles(0, -90, 90, 4.5, None)
    np.testing.assert_allclose(angles, np.arange(-90, 90, 4.5))

    angles = floret.generate_initial_angles(0, -90, 90, 4, None)
    np.testing.assert_allclose(angles, np.arange(-88, 90, 4))

    angles = floret.generate_initial_angles(-4.5, -90, 90, 4.5, None)
    np.testing.assert_allclose(angles, np.arange(-90, 90 - 2 * 4.5, 4.5))

    angles = floret.generate_initial_angles(-10, -90, 90, 4.5, None)
    np.testing.assert_allclose(angles, np.arange(-86.5, 90 - 2 * 10, 4.5))

    angles = floret.generate_initial_angles(-10, -90, 90, None, 40)
    np.testing.assert_allclose(angles, np.arange(-90, 90, 180 / 40))

    angles = floret.generate_initial_angles(-10, -90, 90, None, 41)
    np.testing.assert_allclose(angles, np.arange(-90, 90, 180 / 41))


def test_shuffle_array():
    x = floret.shuffle_array(np.arange(11), 1)
    np.testing.assert_allclose(x, [0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5])

    x = floret.shuffle_array(np.arange(11), 2)
    np.testing.assert_allclose(x, [0, 1, 10, 9, 2, 3, 8, 7, 4, 5, 6])

    x = floret.shuffle_array(np.arange(11), 3)
    np.testing.assert_allclose(x, [0, 1, 2, 10, 9, 8, 3, 4, 5, 7, 6])

    x = floret.shuffle_array(np.arange(11), 4)
    np.testing.assert_allclose(x, [0, 1, 2, 3, 10, 9, 8, 7, 4, 5, 6])

    x = floret.shuffle_array(np.arange(11), 5)
    np.testing.assert_allclose(x, [0, 1, 2, 3, 4, 10, 9, 8, 7, 6, 5])


def test_generate_dose_symmetry_scan():
    angles = np.arange(-90, 90, 4.5)

    order = floret.generate_dose_symmetric_scan(angles, 0)
    a0 = angles[order]

    np.testing.assert_allclose(a0, angles)

    b1 = [
        0.0,
        -4.5,
        4.5,
        -9.0,
        9.0,
        -13.5,
        13.5,
        -18.0,
        18.0,
        -22.5,
        22.5,
        -27.0,
        27.0,
        -31.5,
        31.5,
        -36.0,
        36.0,
        -40.5,
        40.5,
        -45.0,
        45.0,
        -49.5,
        49.5,
        -54.0,
        54.0,
        -58.5,
        58.5,
        -63.0,
        63.0,
        -67.5,
        67.5,
        -72.0,
        72.0,
        -76.5,
        76.5,
        -81.0,
        81.0,
        -85.5,
        85.5,
        -90.0,
    ]

    order = floret.generate_dose_symmetric_scan(angles, 1)
    a1 = angles[order]
    np.testing.assert_allclose(a1, b1)

    b2 = [
        0.0,
        -90.0,
        -4.5,
        85.5,
        4.5,
        -85.5,
        -9.0,
        81.0,
        9.0,
        -81.0,
        -13.5,
        76.5,
        13.5,
        -76.5,
        -18.0,
        72.0,
        18.0,
        -72.0,
        -22.5,
        67.5,
        22.5,
        -67.5,
        -27.0,
        63.0,
        27.0,
        -63.0,
        -31.5,
        58.5,
        31.5,
        -58.5,
        -36.0,
        54.0,
        36.0,
        -54.0,
        -40.5,
        49.5,
        40.5,
        -49.5,
        -45.0,
        45.0,
    ]

    order = floret.generate_dose_symmetric_scan(angles, 2)
    a2 = angles[order]
    np.testing.assert_allclose(a2, b2)

    b3 = [
        0.0,
        -90.0,
        45.0,
        -45.0,
        -4.5,
        85.5,
        -49.5,
        40.5,
        4.5,
        -85.5,
        49.5,
        -40.5,
        -9.0,
        81.0,
        -54.0,
        36.0,
        9.0,
        -81.0,
        54.0,
        -36.0,
        -13.5,
        76.5,
        -58.5,
        31.5,
        13.5,
        -76.5,
        58.5,
        -31.5,
        -18.0,
        72.0,
        -63.0,
        27.0,
        18.0,
        -72.0,
        63.0,
        -27.0,
        -22.5,
        67.5,
        -67.5,
        22.5,
    ]

    order = floret.generate_dose_symmetric_scan(angles, 3)
    a3 = angles[order]
    np.testing.assert_allclose(a3, b3)

    b4 = [
        0.0,
        -90.0,
        45.0,
        -45.0,
        22.5,
        -67.5,
        67.5,
        -22.5,
        -4.5,
        85.5,
        -49.5,
        40.5,
        -27.0,
        63.0,
        -72.0,
        18.0,
        4.5,
        -85.5,
        49.5,
        -40.5,
        27.0,
        -63.0,
        72.0,
        -18.0,
        -9.0,
        81.0,
        -54.0,
        36.0,
        -31.5,
        58.5,
        -76.5,
        13.5,
        9.0,
        -81.0,
        54.0,
        -36.0,
        31.5,
        -58.5,
        76.5,
        -13.5,
    ]

    order = floret.generate_dose_symmetric_scan(angles, 4)
    a4 = angles[order]
    np.testing.assert_allclose(a4, b4)

    b5 = [
        0.0,
        -90.0,
        45.0,
        -45.0,
        22.5,
        -67.5,
        67.5,
        -22.5,
        -13.5,
        76.5,
        -58.5,
        31.5,
        -36.0,
        54.0,
        -81.0,
        9.0,
        -4.5,
        85.5,
        -49.5,
        40.5,
        -27.0,
        63.0,
        -72.0,
        18.0,
        13.5,
        -76.5,
        58.5,
        -31.5,
        36.0,
        -54.0,
        81.0,
        -9.0,
        4.5,
        -85.5,
        49.5,
        -40.5,
        -18.0,
        72.0,
        -63.0,
        27.0,
    ]

    order = floret.generate_dose_symmetric_scan(angles, 5)
    a5 = angles[order]
    np.testing.assert_allclose(a5, b5)


def test_spiral_scan():
    angles = np.arange(-90, 90, 4.5)

    order = floret.generate_spiral_scan(angles, 1)
    a0 = angles[order]
    np.testing.assert_allclose(a0, angles)

    order = floret.generate_spiral_scan(angles, 2)
    a0 = angles[order]
    b0 = np.concatenate([angles[0::2], angles[1::2]])
    np.testing.assert_allclose(a0, b0)

    order = floret.generate_spiral_scan(angles, 4)
    a0 = angles[order]
    b0 = np.concatenate([angles[0::4], angles[1::4], angles[2::4], angles[3::4]])
    np.testing.assert_allclose(a0, b0)


def test_swinging_scan():
    angles = np.arange(-90, 90, 4.5)

    order = floret.generate_swinging_scan(angles, 1)
    a0 = angles[order]
    np.testing.assert_allclose(a0, angles)

    order = floret.generate_swinging_scan(angles, 2)
    a0 = angles[order]
    b0 = np.concatenate([angles[0::2], np.flip(angles[1::2])])
    np.testing.assert_allclose(a0, b0)

    order = floret.generate_swinging_scan(angles, 4)
    a0 = angles[order]
    b0 = np.concatenate(
        [angles[0::4], np.flip(angles[3::4]), angles[1::4], np.flip(angles[2::4])]
    )
    np.testing.assert_allclose(a0, b0)


def test_generate_initial_positions():
    positions = floret.generate_initial_positions(1, 10)
    np.testing.assert_allclose(positions, np.zeros(10))

    positions = floret.generate_initial_positions(2, 10)
    np.testing.assert_allclose(positions, np.tile([0, 0.5], 5))

    positions = floret.generate_initial_positions(5, 10)
    np.testing.assert_allclose(positions, np.tile([0, 0.2, 0.4, 0.6, 0.8], 2))


def test_generate_shifted_positions():
    a0 = np.arange(10)
    p0 = floret.generate_initial_positions(1, 10)
    angles, positions = floret.generate_shifted_positions(a0, p0, 0, 2)
    np.testing.assert_allclose(positions, np.stack([np.zeros(10), np.ones(10)]))


def test_generate_final_order():
    a0 = np.arange(10)
    p0 = floret.generate_initial_positions(1, 10)
    a1, p1 = floret.generate_shifted_positions(a0, p0, 0, 2)

    angles, positions = floret.generate_final_order(
        a1, p1, order_by="angle", interleave_positions=False
    )

    np.testing.assert_allclose(
        positions, np.stack([np.zeros(10), np.ones(10)]).T.flatten()
    )

    angles, positions = floret.generate_final_order(
        a1, p1, order_by="position", interleave_positions=False
    )

    np.testing.assert_allclose(positions.flatten(), p1.flatten())


def test_generate_scan():
    positions, angles = floret.generate_scan(0, -90, 90, 4.5, symmetry=0)
    np.testing.assert_allclose(angles, np.arange(-90, 90, 4.5))

    b5 = [
        0.0,
        -90.0,
        45.0,
        -45.0,
        22.5,
        -67.5,
        67.5,
        -22.5,
        -13.5,
        76.5,
        -58.5,
        31.5,
        -36.0,
        54.0,
        -81.0,
        9.0,
        -4.5,
        85.5,
        -49.5,
        40.5,
        -27.0,
        63.0,
        -72.0,
        18.0,
        13.5,
        -76.5,
        58.5,
        -31.5,
        36.0,
        -54.0,
        81.0,
        -9.0,
        4.5,
        -85.5,
        49.5,
        -40.5,
        -18.0,
        72.0,
        -63.0,
        27.0,
    ]

    _, a5 = floret.generate_scan(0, -90, 90, 4.5, symmetry=5)
    np.testing.assert_allclose(a5, b5)

    _, a0 = floret.generate_scan(0, -90, 90, 4.5, mode="spiral", stepnum=4)
    b0 = np.concatenate([angles[0::4], angles[1::4], angles[2::4], angles[3::4]])
    np.testing.assert_allclose(a0, b0)

    _, a0 = floret.generate_scan(0, -90, 90, 4.5, mode="swinging", stepnum=4)
    b0 = np.concatenate(
        [angles[0::4], np.flip(angles[3::4]), angles[1::4], np.flip(angles[2::4])]
    )
    np.testing.assert_allclose(a0, b0)


def test_command_line():
    floret.command_line.main(["--tilt_angle_step=4.5"])
