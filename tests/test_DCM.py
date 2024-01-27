import icecream
import numpy
import pytest
from numpy import array, allclose, eye
from src.utils import DCM


# Test for axis vector rotation
@pytest.mark.parametrize(
    "direction, expected_dcm", [
        ("north", array([1, 0, 0])),
        ("east", array([0, 1, 0])),
        ("down", array([0, 0, 1]))
    ])
def test_axis_vector(direction, expected_dcm):
    assert allclose(DCM(0, 0, 0).rot_axis(direction), expected_dcm)


# Test for yaw rotation
@pytest.mark.parametrize(
    "yaw, expected_dcm", [
        (90, array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        (-90, array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
        (180, array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
    ])
def test_yaw_rotation(yaw, expected_dcm):
    assert allclose(DCM(yaw, 0, 0).matrix, expected_dcm)


# Test for pitch rotation
@pytest.mark.parametrize(
    "pitch, expected_dcm", [
        (90, array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
        (-90, array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])),
        (180, array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
    ])
def test_pitch_rotation(pitch, expected_dcm):
    assert allclose(DCM(0, pitch, 0).matrix, expected_dcm)


# Test for roll rotation
@pytest.mark.parametrize(
    "roll, expected_dcm", [
        (90, array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
        (180, array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
        (-90, array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    ])
def test_roll_rotation(roll, expected_dcm):
    assert allclose(DCM(0, 0, roll).matrix, expected_dcm)


def test_inverse_rotation():
    dcm = DCM(90, 0,0).matrix
    inverse_dcm = DCM(-90, 0, 0).matrix
    assert allclose(dcm @ inverse_dcm, eye(dcm.shape[0]))


# Test for angle wrapping
@pytest.mark.parametrize(
    "yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll", [
        (45, 45, 45, 45 - 360, 45 - 360, 45 - 360),
        (30, 60, 90, 30 - 360, 60 - 360, 90 - 360),
        (120, 150, 180, 120 - 360, 150 - 360, 180 - 360)
    ])
def test_angle_wrapping(yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll):
    assert allclose(DCM(yaw, pitch, roll).matrix, DCM(modified_yaw, modified_pitch, modified_roll).matrix)


# Test for invalid input type
def test_invalid_input_type():
    with pytest.raises(TypeError):
        DCM("invalid", "invalid", "invalid")

    # Test for invalid axis input


def test_invalid_axis_input():
    with pytest.raises(AssertionError):
        DCM(0, 0, 0).rot_axis("invalid")
