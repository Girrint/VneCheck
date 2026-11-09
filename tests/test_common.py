import unittest
from pathlib import Path

from vnecheck.data.common import (
    deterministic_split,
    parse_filename_identity,
    read_yolo_label,
    source_group_id,
)


class CommonDataTests(unittest.TestCase):
    def test_source_group_id_removes_roboflow_suffix(self) -> None:
        self.assertEqual(
            source_group_id("freshApple-145-_jpg.rf.abc123.jpg"),
            "freshapple-145-",
        )

    def test_parse_filename_identity(self) -> None:
        cases = {
            "freshApple-145-_jpg.rf.abc.jpg": ("fresh", "apple"),
            "rottenTomato-7-_jpg.rf.abc.jpg": ("rotten", "tomato"),
            "apple_f_0635_jpg.rf.abc.jpg": ("fresh", "apple"),
            "apple-r-12_jpg.rf.abc.jpg": ("rotten", "apple"),
            "1027_jpg.rf.abc.jpg": (None, None),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(parse_filename_identity(filename), expected)

    def test_read_yolo_label(self) -> None:
        label = Path(__file__).parent / "fixtures" / "valid_label.txt"
        boxes = read_yolo_label(label)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_id, 3)

    def test_invalid_box_is_rejected(self) -> None:
        label = Path(__file__).parent / "fixtures" / "invalid_label.txt"
        with self.assertRaisesRegex(ValueError, "outside"):
            read_yolo_label(label)

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(deterministic_split("same-source"), deterministic_split("same-source"))


if __name__ == "__main__":
    unittest.main()
