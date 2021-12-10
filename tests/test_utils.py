from src.utils import check_direction, direction_config, is_intersect

# pylint:disable=unexpected-keyword-arg


class TestCheckDirection:
    def test_true(self):
        """Test true case."""
        directions = {
            "right": {"prev_center": [0, 0], "current_center": [20, 0], "expect": True},
            "left": {"prev_center": [10, 0], "current_center": [0, 0], "expect": True},
            "top": {"prev_center": [0, 10], "current_center": [0, 0], "expect": True},
            "bottom": {"prev_center": [0, 0], "current_center": [0, 10], "expect": True},
        }
        for direction_str, args in directions.items():
            expect = args.pop("expect")
            result = check_direction(**args, direction=direction_config[direction_str])
            assert result == expect

    def test_false(self):
        """Test false case."""
        directions = {
            "right": {"prev_center": [0, 0], "current_center": [0, 0], "expect": False},
            # This is right.
            "left": {"prev_center": [0, 0], "current_center": [10, 0], "expect": False},
            # This is bottom.
            "top": {"prev_center": [0, 0], "current_center": [0, 10], "expect": False},
            # This is top.
            "bottom": {"prev_center": [0, 10], "current_center": [0, 0], "expect": False},
        }
        for direction_str, args in directions.items():
            expect = args.pop("expect")
            result = check_direction(**args, direction=direction_config[direction_str])
            assert result == expect

    def test_direction_none(self):
        """Check if always return true when direction is set None."""
        args = [
            {"prev_center": [0, 0], "current_center": [0, 0]},  # No movement.
            {"prev_center": [0, 0], "current_center": [10, 0]},  # Right
            {"prev_center": [10, 0], "current_center": [0, 0]},  # Left.
            {"prev_center": [0, 10], "current_center": [0, 0]},  # Top.
            {"prev_center": [0, 0], "current_center": [0, 10]},  # Bottom.
        ]
        for arg in args:
            # If the direction is None, always return True.
            result = check_direction(**arg, direction=None)
            assert result == True


class TestIsIntersect:
    def test_true(self):
        """Test true case."""
        args = {"A": [10, 0], "B": [10, 30], "C": [0, 10], "D": [30, 0]}
        result = is_intersect(**args)
        assert result == True

    def test_false(self):
        """Test false case."""
        args = {"A": [10, 0], "B": [10, 30], "C": [0, 10], "D": [0, 0]}
        result = is_intersect(**args)
        assert result == False
