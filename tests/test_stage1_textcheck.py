import unittest

from PIL import Image, ImageDraw

from document_reconstruction.model import Box
from document_reconstruction.recognition.coverage import uncovered_text
from document_reconstruction.recognition.types import Word
from document_reconstruction.recognition.wordcheck import check_page_text, normalize_lookalikes


class FakeLexicon:
    def __init__(self, words):
        self.words = {word.lower() for word in words}

    def __bool__(self):
        return True

    def known(self, token):
        return token.lower() in self.words or all(part.lower() in self.words for part in token.split("-") if part)

    def near_russian(self, token):
        return token.lower() in {"созданин"}


LEXICON = FakeLexicon("в соответствии с пунктом закона создании совета республики года ондан ары".split())


def line(texts, *, y=100, line_id=(1,), confidence=0.95, x=50, step=90):
    return [Word(text, Box(x + i * step, y, x + i * step + 70, y + 20), confidence, line_id=line_id) for i, text in enumerate(texts)]


class WordCheckTests(unittest.TestCase):
    def test_dictionary_text_passes(self):
        result = check_page_text(line(["В", "соответствии", "с", "пунктом", "закона"]), LEXICON)
        self.assertEqual(result.errors, 0)

    def test_misread_word_in_sentence_is_an_error(self):
        result = check_page_text(line(["В", "соответствии", "с", "созданин", "закона"]), LEXICON)
        self.assertEqual(result.errors, 1)

    def test_isolated_unreadable_insert_becomes_empty_field(self):
        words = line(["закона"]) + [Word("oin", Box(600, 95, 640, 125), 0.5, line_id=(1,), source_region="ocr_unstable")]
        result = check_page_text(words, LEXICON)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.blank_fields, 1)
        self.assertEqual(set(result.words[-1].text), {"_"})

    def test_unstable_word_of_print_size_is_an_error_not_a_field(self):
        # "й." misread as "ï." is printed text: it must not silently become an empty line.
        words = line(["закона"]) + [Word("ïx", Box(600, 100, 640, 120), 0.5, line_id=(1,), source_region="ocr_unstable")]
        result = check_page_text(words, LEXICON)
        self.assertEqual((result.errors, result.blank_fields), (1, 0))

    def test_stably_read_isolated_word_is_not_blanked(self):
        words = line(["закона"]) + [Word("oin", Box(600, 100, 640, 120), 0.5, line_id=(2,))]
        result = check_page_text(words, LEXICON)
        self.assertEqual((result.errors, result.blank_fields), (1, 0))

    def test_names_and_national_forms_are_unverified_not_errors(self):
        words = line(["Орумбаевна", "жәшігіңіз", "закона"])
        result = check_page_text(words, LEXICON)
        self.assertEqual(result.errors, 0)
        self.assertEqual(result.unverified, 2)

    def test_weak_numbers_are_not_trusted(self):
        words = line(["года", "1992", "закона"], confidence=0.95)
        words[1].confidence = 0.5
        self.assertEqual(check_page_text(words, LEXICON).errors, 1)

    def test_line_end_hyphenation_is_checked_as_one_word(self):
        words = line(["республики", "сов-"], line_id=(1,)) + line(["ета", "года"], y=130, line_id=(2,))
        self.assertEqual(check_page_text(words, LEXICON).errors, 0)

    def test_latin_letter_in_kazakh_line_is_an_error(self):
        # On a Kazakh page a Latin K may stand for Қ: it is not silently replaced.
        words = line(["Қызылорда", "көшесі", "Әлия", "Нұрланқызы", "K,"])
        self.assertEqual(check_page_text(words, LEXICON).errors, 1)

    def test_russian_page_lookalikes_and_number_sign_are_normalized(self):
        words = line(["года", "Ne", "12", "закона", "2024r.", "O", "совета"])
        result = check_page_text(words, LEXICON)
        self.assertEqual([word.text for word in result.words], ["года", "№", "12", "закона", "2024 г.", "О", "совета"])

    def test_bullet_read_as_letter_becomes_bullet(self):
        words = [Word("e", Box(50, 100, 58, 120), 0.4, line_id=(1,))] + line(["закона", "года"], x=90)
        result = check_page_text(words, LEXICON)
        self.assertEqual(result.words[0].text, "•")
        self.assertEqual(result.errors, 0)

    def test_lookalike_letters_are_normalized_to_the_word_alphabet(self):
        words = line(["Pеспублика", "Ə."])
        normalize_lookalikes(words)
        self.assertEqual([word.text for word in words], ["Республика", "Ә."])


class CoverageTests(unittest.TestCase):
    def _page(self):
        image = Image.new("RGB", (1000, 1400), "white")
        draw = ImageDraw.Draw(image)
        for row in range(3):
            for column in range(8):
                draw.rectangle((100 + column * 30, 200 + row * 40, 118 + column * 30, 220 + row * 40), fill="black")
        return image

    def test_unread_text_line_is_reported(self):
        image = self._page()
        words = [Word("x" * 8, Box(100, 200, 330, 220), 0.95, line_id=(0,))]
        coverage = uncovered_text(image, words, [], [], width=1000, height=1400)
        self.assertEqual(len(coverage.clusters), 2)

    def test_fully_read_page_has_no_lost_text(self):
        image = self._page()
        words = [Word("x" * 8, Box(100, 200 + row * 40, 330, 220 + row * 40), 0.95, line_id=(row,)) for row in range(3)]
        self.assertFalse(uncovered_text(image, words, [], [], width=1000, height=1400).lost_text)


if __name__ == "__main__":
    unittest.main()
