import unittest
from data.candles_tink import split_time_period


class TestSplitTimePeriods(unittest.TestCase):

    def test_edgeI(self):
        from_iso = "2026-01-01"
        to_iso = "2026-01-01"
        with self.assertRaises(ValueError):
            periods = split_time_period(from_iso, to_iso, 1)

    def test_edgeII(self):
        from_iso = "2026-01-01"
        to_iso = "2026-01-02"
        with self.assertRaises(ValueError):
            periods = split_time_period(from_iso, to_iso, -1)

    def test_edgeIII(self):
        from_iso = "2026-01-01"
        to_iso = "2026-13-12"
        with self.assertRaises(ValueError):
            periods = split_time_period(from_iso, to_iso, 1)

    def test_edgeIV(self):
        from_iso = "2026-01-01"
        to_iso = "2026-01-02"
        periods = split_time_period(from_iso, to_iso, 1)
        self.assertEqual(["2026-01-01T00:00:00", "2026-01-02T00:00:00"], periods)


if __name__ == "__main__":
    unittest.main()
