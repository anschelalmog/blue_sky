from src.utils import euler_to_dcm
from src.pinpoint_calc import jac_east, jac_north
from numpy import array, array_equal, dot, allclose
import pytest


@pytest.mark.parametrize(
    "direction, expected_dcm",
    [("north", array([1, 0, 0])),
     ("east", array([0, 1, 0])),
     ("down", array([0, 0, 1]))
     ])
def test_euler_to_dcm(direction, expected_dcm):
    dcm = euler_to_dcm(direction, 0, 0, 0)
    assert array_equal(dcm, expected_dcm)


@pytest.mark.parametrize(
    "yaw, expected_dcm",
    [(90, array([0, 1, 0])),
     (-90, array([0, -1, 0])),
     (180, array([-1, 0, 0]))
     ])
def test_yaw_rotation(yaw, expected_dcm):
    dcm = euler_to_dcm('north', yaw, 0, 0)
    assert allclose(dcm, expected_dcm)


@pytest.mark.parametrize(
    "pitch,expected_dcm",
    [(90, array([0, 0, -1])),
     (-90, array([0, 0, 1])),
     (180, array([-1, 0, 0]))
     ])
def test_pitch_rotation(pitch, expected_dcm):
    dcm = euler_to_dcm('north', 0, pitch, 0)
    assert allclose(dcm, expected_dcm)


@pytest.mark.parametrize(
    "rot_ax, roll, expected_dcm",
    [('north', 90, array([1, 0, 0])),
     ('north', 180, array([1, 0, 0])),
     ('east', 180, array([0, -1, 0])),
     ('down', 180, array([0, 0, -1]))
    ])
def test_roll_rotation(rot_ax, roll, expected_dcm):
    dcm = euler_to_dcm(rot_ax, 0, 0, roll)
    assert allclose(dcm, expected_dcm)


#todo: ask Oshra if the projection on the axis should after to DCM function
#      and not inside to function

# @pytest.mark.parametrize(
#     "yaw, pitch, roll, expected",
#     [(0,  0, 0, array([1, 0, 0])),
#      (90, 0, 0, array([1, 0, 0])),
#      (0, 90, 0, array([1, 0, 0])),
#      (0,  0, 180, array([1, 0, 0]))
#      ])
# def test_inverse_rotation(yaw, pitch, roll, expected):
#     dcm = euler_to_dcm("north", yaw, pitch, roll)
#     inv_dcm = euler_to_dcm("north", -yaw, -pitch, -roll)
#     result = dcm * inv_dcm
#     assert allclose(result, expected)

def test_inverse_rotation():
    yaw, pitch, roll = 0, 0, 0
    dcm = euler_to_dcm("north", yaw, pitch, roll)
    inv_dcm = euler_to_dcm("north", -yaw, -pitch, -roll)
    result = dcm * inv_dcm
    expected = array([1, 0, 0])
    assert allclose(result, expected)

@pytest.mark.parametrize(
    "yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll",
    [(45, 45, 45, 45 - 360, 45 - 360, 45 - 360),
     (30, 60, 90, 30 - 360, 60 - 360, 90 - 360),
     (120, 150, 180, 120 - 360, 150 - 360, 180 - 360)
     ])
def test_angle_wrapping(yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll):
    dcm_1 = euler_to_dcm("east", yaw, pitch, -roll)
    dcm_2 = euler_to_dcm("east", modified_yaw, modified_pitch, -modified_roll)
    assert allclose(dcm_1, dcm_2)


def test_invalid_input_type():
    with pytest.raises(AssertionError):
        euler_to_dcm([1, 0, 0], [45], None, 0)


def test_invalid_axis_input():
    with pytest.raises(AssertionError):
        euler_to_dcm("invalid", 0, 0, 0)
