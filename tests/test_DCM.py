import pytest
import unittest
from numpy import array, allclose, eye
from archive.archive_2 import DCM


class TestDCM(unittest.TestCase):
    @pytest.mark.parametrize(
        "direction, expected_dcm", [
            ("north", array([1, 0, 0])),
            ("east", array([0, 1, 0])),
            ("down", array([0, 0, 1]))
        ])
    def test_axis_vector(self, direction, expected_dcm):
        assert allclose(DCM(0, 0, 0)._rot_axis(direction), expected_dcm)

    @pytest.mark.parametrize(
        "yaw, expected_dcm", [
            (90, array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
            (-90, array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
            (180, array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
        ])
    def test_yaw_rotation(self, yaw, expected_dcm):
        assert allclose(DCM(yaw, 0, 0).matrix, expected_dcm)

    @pytest.mark.parametrize(
        "pitch, expected_dcm", [
            (90, array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
            (-90, array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])),
            (180, array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        ])
    def test_pitch_rotation(self, pitch, expected_dcm):
        assert allclose(DCM(0, pitch, 0).matrix, expected_dcm)

    @pytest.mark.parametrize(
        "roll, expected_dcm", [
            (90, array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
            (180, array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
            (-90, array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
        ])
    def test_roll_rotation(self, roll, expected_dcm):
        assert allclose(DCM(0, 0, roll).matrix, expected_dcm)

    def test_inverse_rotation(self):
        dcm = DCM(90, 0, 0).matrix
        inverse_dcm = DCM(-90, 0, 0).matrix
        assert allclose(dcm @ inverse_dcm, eye(dcm.shape[0]))

    @pytest.mark.parametrize(
        "yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll", [
            (45, 45, 45, 45 - 360, 45 - 360, 45 - 360),
            (30, 60, 90, 30 - 360, 60 - 360, 90 - 360),
            (120, 150, 180, 120 - 360, 150 - 360, 180 - 360)
        ])
    def test_angle_wrapping(self, yaw, pitch, roll, modified_yaw, modified_pitch, modified_roll):
        assert allclose(DCM(yaw, pitch, roll).matrix, DCM(modified_yaw, modified_pitch, modified_roll).matrix)

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            DCM("invalid", "invalid", "invalid")

    def test_invalid_axis_input(self):
        with pytest.raises(AssertionError):
            DCM(0, 0, 0)._rot_axis("invalid")
