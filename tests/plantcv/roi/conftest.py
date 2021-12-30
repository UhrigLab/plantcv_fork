import pytest
import os
import matplotlib

# Disable plotting
matplotlib.use("Template")


class RoiTestData:
    def __init__(self):
        """Initialize simple variables."""
        # Test data directory
        self.datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata")
        # RGB image
        self.small_rgb_img = os.path.join(self.datadir, "setaria_small_rgb_img.png")
        # Gray image
        self.small_gray_img = os.path.join(self.datadir, "setaria_small_gray_img.png")


@pytest.fixture(scope="session")
def roi_test_data():
    return RoiTestData()
