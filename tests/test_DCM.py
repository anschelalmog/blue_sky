from numpy import array, array_equal, dot, allclose
from src.utils import euler_to_dcm
from icecream import ic
import pytest


def test_zero_angles_north():
    dcm = euler_to_dcm("north", 0, 0, 0)
    assert array_equal(dcm, array([1, 0, 0]))


def test_zero_angles_east():
    dcm = euler_to_dcm("east", 0, 0, 0)
    assert array_equal(dcm, array([0, 1, 0]))


def test_zero_angles_down():
    dcm = euler_to_dcm("down", 0, 0, 0)
    assert array_equal(dcm, array([0, 0, 1]))


@pytest.mark.parametrize(
    "roll_angle, expected_dcm",
    [(90, array([0, -1, 0])),
     (-90, array([0, 1, 0])),
     (180, array([-1, 0, 0]))
    ])
def test_yaw_rotation(roll_angle, expected_dcm):
    dcm = euler_to_dcm('north', roll_angle, 0, 0)
    ic(dcm)
    assert allclose(dcm, expected_dcm)


@pytest.mark.parametrize(
    "pitch_angle,expected_dcm",
    [(90, array([0, 0, -1])),
     (-90, array([0, 0, 1])),
     (180, array([-1, 0, 0]))
     ])
def test_pitch_rotation(pitch_angle, expected_dcm):
    dcm = euler_to_dcm('north', 0, pitch_angle, 0)
    assert allclose(dcm, expected_dcm)


def test_roll_rotation():
    dcm = euler_to_dcm("down", 0, 0, 90)
    expected = array([0, -1, 0])
    assert allclose(dcm, expected)


def test_inverse_rotation():
    yaw, pitch, roll = 0, 0, 0
    dcm = euler_to_dcm("north", yaw, pitch, roll)
    inv_dcm = euler_to_dcm("north", -yaw, -pitch, -roll)
    result = dot(dcm, inv_dcm)
    expected = array([1, 0, 0])
    assert allclose(result, expected)


def test_angle_wrapping():
    yaw, pitch, roll = 45, 45, 45
    dcm_1 = euler_to_dcm("east", yaw, pitch, -roll)
    dcm_2 = euler_to_dcm("east", yaw - 360, pitch - 360, -(roll - 360))
    assert allclose(dcm_1, dcm_2)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        euler_to_dcm([1, 0, 0], [45], None, 0)


def test_invalid_axis_input():
    with pytest.raises(AssertionError):
        euler_to_dcm("invalid", 0, 0, 0)
