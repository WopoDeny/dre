import unittest

import numpy as np
from PIL import Image, ImageDraw

from document_reconstruction.model import Box, StructuralIssue
from document_reconstruction.recognition.marks import color_marks, printed_color_words, resolve_layers, split_ink
from document_reconstruction.recognition.regions import blank_obscured_words
from document_reconstruction.recognition.types import Word


def page():
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    for row in range(10):
        draw.rectangle((100, 100 + row * 40, 700, 118 + row * 40), fill=(20, 20, 20))
    return image, draw


class InkSeparationTests(unittest.TestCase):
    def test_black_text_under_a_blue_stamp_stays_in_black_layer(self):
        image, draw = page()
        draw.ellipse((300, 60, 600, 360), outline=(40, 60, 200), width=10)
        layers = split_ink(image)
        self.assertTrue(layers.diagnostics["ink_separation"])
        black = np.asarray(layers.black.convert("L"))
        self.assertLess(black[108, 150], 60)             # black text kept
        self.assertGreater(black[210, 305], 150)         # stamp ring removed where it crosses paper
        self.assertLess(np.asarray(layers.color)[210, 305], 150)

    def test_color_cast_page_is_not_separated(self):
        image = Image.new("RGB", (800, 800), (240, 220, 180))
        ImageDraw.Draw(image).rectangle((100, 100, 700, 700), fill=(120, 80, 40))
        self.assertIsNone(split_ink(image).color)

    def test_printed_colored_line_is_text_and_handwriting_is_not(self):
        printed = [Word("ПРИКАЗ", Box(0, 0, 50, 10), 0.9, line_id=(1,)), Word("города", Box(60, 0, 90, 10), 0.85, line_id=(1,))]
        handwriting = [Word("7f/", Box(0, 30, 30, 40), 0.3, line_id=(2,)), Word("2T", Box(40, 30, 60, 40), 0.4, line_id=(2,))]
        self.assertEqual(printed_color_words(printed + handwriting), printed)

    def test_black_ghost_of_colored_word_is_dropped_but_text_under_stamp_is_kept(self):
        image, draw = page()
        draw.rectangle((100, 600, 300, 618), fill=(60, 80, 220))    # colored handwriting only
        mask = split_ink(image).color_mask
        ghost = Word("x7", Box(100, 600, 300, 618), 0.3)
        under_stamp = Word("текст", Box(100, 100, 300, 118), 0.4)
        self.assertEqual(resolve_layers(image, mask, [ghost, under_stamp], [], width=1000, height=1400), [under_stamp])


class ColorMarkTests(unittest.TestCase):
    def _marks(self, image, words=()):
        layers = split_ink(image)
        return color_marks(image, layers.color_mask, list(words), index=0, width=image.width, height=image.height)

    def test_stamp_is_picture_and_small_handwriting_is_blank_field(self):
        image, draw = page()
        text = [Word("слово", Box(100, 100 + row * 40, 700, 118 + row * 40), 0.95, line_id=(row,)) for row in range(10)]
        draw.ellipse((300, 800, 560, 1060), outline=(40, 60, 200), width=8)
        draw.line((150, 700, 190, 680), fill=(30, 50, 190), width=3)
        draw.line((190, 680, 220, 705), fill=(30, 50, 190), width=3)
        graphics, blanks, _ = self._marks(image, text)
        self.assertEqual([graphic.role for graphic in graphics], ["seal"])
        self.assertEqual(len(blanks), 1)
        self.assertTrue(set(blanks[0].text) == {"_"})

    def test_watermark_and_vertical_margin_text_are_ignored(self):
        image, draw = page()
        draw.ellipse((300, 800, 560, 1060), fill=(200, 215, 250))          # pale watermark
        draw.rectangle((15, 300, 25, 900), fill=(40, 60, 200))             # vertical margin note
        graphics, blanks, diagnostics = self._marks(image)
        self.assertEqual(graphics, [])
        self.assertEqual(blanks, [])
        self.assertEqual(diagnostics["color_margin_ignored"], 2)


class ObscuredWordTests(unittest.TestCase):
    def test_isolated_word_under_signature_becomes_blank_but_word_in_sentence_does_not(self):
        title = Word("управления", Box(100, 500, 300, 520), 0.96, line_id=(7,))
        under_signature = Word("сЕ", Box(700, 500, 760, 520), 0.3, line_id=(7,))
        in_sentence = Word("тра", Box(320, 500, 360, 520), 0.3, line_id=(7,))
        issues = [StructuralIssue("unresolved_graphic_text_overlap", "", "error", 0.3, word.box) for word in (under_signature, in_sentence)]
        words, remaining, blanked = blank_obscured_words([title, in_sentence, under_signature], issues)
        self.assertEqual(blanked, 1)
        self.assertEqual(words[2].source_region, "blank_field")
        self.assertEqual([issue.box for issue in remaining], [in_sentence.box])


if __name__ == "__main__":
    unittest.main()
