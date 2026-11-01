import unittest

from src.track_video import make_detection, scaled_size


class TestVideoHelpers(unittest.TestCase):
    def test_downscale_hd(self):
        self.assertEqual(scaled_size(1920, 1080, 640), (640, 360))

    def test_do_not_upscale(self):
        self.assertEqual(scaled_size(320, 240, 640), (320, 240))

    def test_even_dimensions(self):
        width, height = scaled_size(641, 359, 640)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            scaled_size(0, 480)

    def test_xyxy_to_deepsort(self):
        self.assertEqual(
            make_detection(10, 20, 50, 80, 0.9, "person"),
            ([10.0, 20.0, 40.0, 60.0], 0.9, "person"),
        )

    def test_invalid_box_skipped(self):
        self.assertIsNone(make_detection(50, 20, 10, 80, 0.9, "person"))


if __name__ == "__main__":
    unittest.main()
