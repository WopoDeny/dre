"""Document structure and Word package regression tests."""

import io
import json
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from document_reconstruction.model import Box, Cell, Document, Graphic, InlineRun, Page, Paragraph, StructuralIssue, Table
from document_reconstruction.quality import NS, verify
from document_reconstruction.recognition.languages import direction, profile, script_counts
from document_reconstruction.recognition.types import RecognizedPage, Word
from document_reconstruction.reconstruction.docx import reconstruct
from document_reconstruction.understanding import understand_page
from document_reconstruction.core import EngineError


def line(text, y, *, x=48, end=490, size=11, number=0, bold=False, italic=False, confidence=1.0, region="test"):
    return Word(text, Box(x, y, end, y + 12), confidence=confidence, size=size, bold=bold, italic=italic, line_id=(number,), source_page=0, source_region=region)


class UnderstandingTests(unittest.TestCase):
    def test_visual_lines_become_one_paragraph(self):
        page = understand_page(RecognizedPage(0, 595, 842, [line("The document contains a long", 100, number=0), line("sentence continued on another line", 116, number=1), line("with its final words here.", 132, number=2)]))
        self.assertEqual(len(page.blocks), 1)
        self.assertEqual(page.blocks[0].source_lines, 3)
        self.assertNotIn("\n", page.blocks[0].text)

    def test_separate_paragraphs_and_heading(self):
        page = understand_page(RecognizedPage(0, 595, 842, [line("Review", 60, size=18, number=0), line("First paragraph.", 100, number=1), line("Second paragraph.", 145, number=2)]))
        self.assertEqual([block.kind for block in page.blocks], ["heading", "paragraph", "paragraph"])

    def test_column_reading_order(self):
        words = []
        for i in range(3):
            words.extend([line(f"Left {i}", 100 + 16 * i, end=230, number=i), line(f"Right {i}", 100 + 16 * i, x=340, end=540, number=i + 10)])
        page = understand_page(RecognizedPage(0, 595, 842, words))
        # Side-by-side text columns become one borderless layout table, read column by column.
        self.assertEqual(len(page.blocks), 1)
        self.assertTrue(page.blocks[0].borderless)
        self.assertEqual([cell.text for cell in page.blocks[0].cells], ["Left 0 Left 1 Left 2", "Right 0 Right 1 Right 2"])

    def test_four_column_mixed_script_reading_order_with_spanning_heading(self):
        words = [line("Quarterly review", 55, x=45, end=550, size=18, number=90)]
        columns = [
            (45, 150, ["Latin A", "Latin B", "Latin C"]),
            (180, 285, ["Кирилл A", "Кирилл B", "Кирилл C"]),
            (315, 420, ["العربية أ", "العربية ب", "العربية ج"]),
            (450, 555, ["中文甲", "中文乙", "中文丙"]),
        ]
        number = 0
        for left, right, values in columns:
            for row, text in enumerate(values):
                words.append(line(text, 100 + 18 * row, x=left, end=right, number=number))
                number += 1
        page = understand_page(RecognizedPage(0, 595, 842, words))
        self.assertEqual(page.blocks[0].text, "Quarterly review")
        self.assertEqual(
            [cell.text for cell in page.blocks[1].cells],
            [
                "Latin A Latin B Latin C",
                "Кирилл A Кирилл B Кирилл C",
                "العربية أ العربية ب العربية ج",
                "中文甲中文乙中文丙",
            ],
        )
        self.assertEqual([cell.column for cell in page.blocks[1].cells], [0, 1, 2, 3])

    def test_lists_do_not_merge(self):
        page = understand_page(RecognizedPage(0, 595, 842, [line("1. First item", 100, number=0), line("2. Second item", 116, number=1)]))
        self.assertEqual([block.kind for block in page.blocks], ["list_item", "list_item"])

    def test_table_text_is_not_duplicated(self):
        table = Table(Box(40, 90, 520, 200), 0, 1, 1, [Cell("Cell value", 0, 0)])
        page = understand_page(RecognizedPage(0, 595, 842, [line("Cell value", 100)], tables=[table]))
        self.assertEqual(len(page.blocks), 1)
        self.assertIsInstance(page.blocks[0], Table)


    def test_list_groups_and_nesting_are_explicit_from_indentation(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("1. Parent item", 100, x=48, end=300, number=0),
            line("a. Nested item", 120, x=72, end=300, number=1),
            line("2. Second parent", 140, x=48, end=300, number=2),
            line("Body paragraph.", 180, x=48, end=220, number=3),
            line("1. New list", 220, x=48, end=260, number=4),
        ]))
        items = [block for block in page.blocks if isinstance(block, Paragraph) and block.kind == "list_item"]
        self.assertEqual([item.level for item in items], [0, 1, 0, 0])
        self.assertIsNotNone(items[0].list_id)
        self.assertEqual([item.list_id for item in items[:3]], [items[0].list_id] * 3)
        self.assertNotEqual(items[0].list_id, items[3].list_id)

    def test_table_cell_runs_are_enriched_only_when_owned_text_matches(self):
        cell = Cell("Bold value", 0, 0, box=Box(40, 90, 280, 140), source_region="native_table")
        table = Table(Box(40, 90, 280, 140), 0, 1, 1, [cell], source_region="native_table")
        words = [
            Word("Bold", Box(55, 105, 95, 118), bold=True, size=12, line_id=(1,), source_page=0, source_region="native_pdf"),
            Word("value", Box(100, 105, 145, 118), italic=True, size=11, line_id=(1,), source_page=0, source_region="native_pdf"),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words, tables=[table]))
        normalized = page.blocks[0]
        self.assertIsInstance(normalized, Table)
        result = normalized.cells[0]
        self.assertEqual("".join(run.text for run in result.runs), result.text)
        self.assertTrue(result.runs[0].bold)
        self.assertTrue(result.runs[1].italic)
        self.assertEqual({run.source_region for run in result.runs}, {"native_pdf"})
        self.assertEqual(page.blocks[0].cells[0].text, "Bold value")

    def test_table_cell_run_mismatch_exposes_issue_without_replacing_text(self):
        cell = Cell("Canonical value", 0, 0, box=Box(40, 90, 280, 140), source_region="native_table")
        table = Table(Box(40, 90, 280, 140), 0, 1, 1, [cell], source_region="native_table")
        words = [Word("Mismatch", Box(55, 105, 125, 118), line_id=(1,), source_page=0, source_region="native_pdf")]
        page = understand_page(RecognizedPage(0, 595, 842, words, tables=[table]))
        result = page.blocks[0].cells[0]
        self.assertEqual(result.text, "Canonical value")
        self.assertEqual(result.runs, [])
        self.assertIn("table_cell_run_mismatch", [issue.code for issue in result.issues])
        self.assertEqual(page.blocks[0].cells[0].text, "Canonical value")


    def test_adjacent_short_fields_are_not_over_merged(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("Account", 100, end=155, number=0),
            line("Approved", 116, end=165, number=1),
            line("Reference:", 150, end=125, number=2),
            line("A-42", 166, end=95, number=3),
        ]))
        self.assertEqual([block.text for block in page.blocks], ["Account", "Approved", "Reference:", "A-42"])

    def test_heading_and_wrapped_list_items_keep_boundaries(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("REQUIREMENTS", 60, end=260, size=12, bold=True, number=0),
            line("1. First item continues across", 100, end=475, number=1),
            line("the next visual line without splitting.", 116, x=64, end=470, number=2),
            line("2. Second item", 136, end=250, number=3),
        ]))
        self.assertEqual([block.kind for block in page.blocks], ["heading", "list_item", "list_item"])
        self.assertEqual(page.blocks[1].source_lines, 2)
        self.assertEqual(page.blocks[1].list_marker, "1.")
        self.assertTrue(page.blocks[1].list_ordered)
        self.assertIn("next visual line", page.blocks[1].text)


    def test_first_line_indent_is_relative_to_text_column_not_page_origin(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("Indented first line continues across the available width", 100, x=72, end=500, number=0),
            line("with a flush continuation on the next visual line.", 116, x=48, end=500, number=1),
        ]))
        self.assertEqual(len(page.blocks), 1)
        paragraph = page.blocks[0]
        self.assertAlmostEqual(paragraph.first_line_indent, 24.0, places=1)
        self.assertEqual(paragraph.alignment, "justify")

    def test_hyphenation_is_conservative(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("The inter-", 100, end=500, number=0),
            line("national standard is ISO-", 116, end=500, number=1),
            line("9001 and uses state-of-", 132, end=500, number=2),
            line("the-art controls.", 148, end=500, number=3),
        ]))
        self.assertEqual(len(page.blocks), 1)
        self.assertEqual(page.blocks[0].text, "The international standard is ISO-9001 and uses state-of-the-art controls.")
        self.assertEqual("".join(run.text for run in page.blocks[0].runs), page.blocks[0].text)

    def test_arabic_logical_order_with_latin_numbers_and_punctuation(self):
        words = [
            Word("البند", Box(430, 100, 480, 114), line_id=(1,), source_page=0, source_region="ocr", size=11),
            Word("12،", Box(390, 100, 420, 114), line_id=(1,), source_page=0, source_region="ocr", size=11),
            Word("الإصدار", Box(315, 100, 380, 114), line_id=(1,), source_page=0, source_region="ocr", size=11),
            Word("A-3.", Box(270, 100, 305, 114), line_id=(1,), source_page=0, source_region="ocr", size=11),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words))
        paragraph = page.blocks[0]
        self.assertEqual(paragraph.text, "البند 12، الإصدار A-3.")
        self.assertEqual(paragraph.direction, "rtl")
        self.assertEqual([run.source_region for run in paragraph.runs], ["ocr", "ocr", "ocr", "ocr"])

    def test_chinese_punctuation_and_visual_line_joining_add_no_western_spaces(self):
        page = understand_page(RecognizedPage(0, 595, 842, [
            line("这是第一行，", 100, end=500, number=0),
            line("继续内容。", 116, end=500, number=1),
        ]))
        self.assertEqual(page.blocks[0].text, "这是第一行，继续内容。")
        self.assertNotIn("， ", page.blocks[0].text)

    def test_chinese_adjacent_latin_numbers_do_not_gain_western_spaces(self):
        words = [
            Word("文档编号", Box(48, 100, 120, 114), line_id=(1,), source_page=0),
            Word("2026-A17，金额", Box(121, 100, 240, 114), line_id=(1,), source_page=0),
            Word("123.45", Box(241, 100, 295, 114), line_id=(1,), source_page=0),
            Word("元。", Box(296, 100, 330, 114), line_id=(1,), source_page=0),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words))
        self.assertEqual(page.blocks[0].text, "文档编号2026-A17，金额123.45元。")
        self.assertNotIn("编号 2026", page.blocks[0].text)
        self.assertNotIn("45 元", page.blocks[0].text)


    def test_inline_runs_preserve_basic_style_and_provenance(self):
        words = [
            Word("Policy", Box(48, 100, 90, 114), bold=True, size=12, line_id=(1,), source_page=0, source_region="native_pdf"),
            Word("text", Box(96, 100, 125, 114), italic=True, size=11, line_id=(1,), source_page=0, source_region="native_pdf"),
            Word("العربية", Box(132, 100, 190, 114), italic=True, size=11, line_id=(1,), source_page=0, source_region="native_pdf"),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words))
        paragraph = page.blocks[0]
        self.assertEqual(paragraph.text, "Policy text العربية")
        self.assertEqual("".join(run.text for run in paragraph.runs), paragraph.text)
        self.assertTrue(paragraph.runs[0].bold)
        self.assertTrue(paragraph.runs[1].italic)
        self.assertEqual(paragraph.runs[-1].script, "Arabic")
        self.assertEqual({ref.region for ref in paragraph.provenance}, {"native_pdf"})

    def test_table_owns_cell_text_exactly_once_and_preserves_cell_geometry(self):
        cell_box = Box(40, 90, 280, 140)
        table = Table(Box(40, 90, 520, 190), 0, 2, 2, [
            Cell("Alpha", 0, 0, box=cell_box),
            Cell("Beta", 0, 1, box=Box(280, 90, 520, 140)),
            Cell("Gamma", 1, 0, column_span=2, box=Box(40, 140, 520, 190)),
        ])
        words = [line("Outside paragraph", 60, end=400, number=0), line("Alpha", 105, x=60, end=120, number=1), line("Beta", 105, x=320, end=370, number=2)]
        page = understand_page(RecognizedPage(0, 595, 842, words, tables=[table]))
        self.assertEqual([type(block).__name__ for block in page.blocks], ["Paragraph", "Table"])
        self.assertEqual(page.blocks[0].text, "Outside paragraph")
        self.assertEqual([cell.text for cell in page.blocks[1].cells], ["Alpha", "Beta", "Gamma"])
        self.assertEqual(page.blocks[1].cells[0].box, cell_box)
        self.assertEqual(page.blocks[1].cells[0].source_page, 0)
        self.assertEqual(page.blocks[1].cells[2].column_span, 2)
        self.assertEqual([row.index for row in page.blocks[1].row_items], [0, 1])
        self.assertEqual([cell.text for cell in page.blocks[1].row_items[0].cells], ["Alpha", "Beta"])
        self.assertEqual(page.blocks[1].cells[0].source_region, "page")
        self.assertEqual(page.blocks[0].text.count("Alpha"), 0)

    def test_graphic_is_separate_while_overlapping_body_text_remains_editable(self):
        image = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(image, format="PNG")
        graphic = Graphic(Box(220, 112, 320, 150), 0, image.getvalue(), description="Isolated document mark", role="stamp", source_region="raster_graphic")
        words = [
            line("Text before the graphic occupies a full logical line", 96, end=520, number=0),
            line("inside mark", 120, x=235, end=300, number=1),
            line("Text after the graphic starts a new paragraph", 154, end=500, number=2),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words, graphics=[graphic]))
        self.assertEqual([type(block).__name__ for block in page.blocks], ["Paragraph", "Graphic", "Paragraph", "Paragraph"])
        self.assertEqual(page.blocks[1].role, "stamp")
        paragraph_text = " ".join(block.text for block in page.blocks if isinstance(block, Paragraph))
        self.assertIn("inside mark", paragraph_text)
        self.assertIn("Text after the graphic", paragraph_text)

    def test_columns_and_normalization_metadata_are_explicit(self):
        words = []
        for i in range(3):
            words.extend([line(f"Left {i}", 100 + 16 * i, end=230, number=i), line(f"Right {i}", 100 + 16 * i, x=340, end=540, number=i + 10)])
        page = understand_page(RecognizedPage(0, 595, 842, words, source_rotation=90))
        self.assertEqual(page.blocks[0].columns, 2)
        self.assertEqual(page.source_rotation, 90)
        self.assertIsNotNone(page.margin_left)
        self.assertIsNotNone(page.margin_right)

    def test_low_confidence_structure_exposes_issue_instead_of_inventing_certainty(self):
        page = understand_page(RecognizedPage(0, 595, 842, [line("Uncertain source text", 100, end=500, number=0, confidence=0.35)]))
        self.assertEqual(page.issues[0].code, "low_text_confidence")
        self.assertLess(page.blocks[0].confidence, 0.5)


    def test_non_region_text_is_neither_dropped_nor_duplicated(self):
        words = [
            Word("Every", Box(48, 100, 85, 114), line_id=(1,), source_page=0),
            Word("source", Box(90, 100, 135, 114), line_id=(1,), source_page=0),
            Word("token", Box(140, 100, 178, 114), line_id=(1,), source_page=0),
            Word("survives.", Box(183, 100, 245, 114), line_id=(1,), source_page=0),
            Word("Second", Box(48, 145, 95, 159), line_id=(2,), source_page=0),
            Word("paragraph.", Box(100, 145, 170, 159), line_id=(2,), source_page=0),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words))
        model_text = " ".join(block.text for block in page.blocks if isinstance(block, Paragraph))
        for token in ("Every", "source", "token", "survives.", "Second", "paragraph."):
            self.assertEqual(model_text.count(token), 1)

    def test_model_integrity_detects_text_geometry_provenance_and_span_contradictions(self):
        broken = Document([Page(0, 200, 200, [
            Paragraph(
                "Expected",
                Box(10, 10, 90, 30),
                0,
                runs=[InlineRun("Different", Box(10, 10, 95, 30), 1)],
            ),
            Table(
                Box(10, 50, 190, 120),
                0,
                1,
                2,
                [
                    Cell("A", 0, 0, column_span=2, box=Box(10, 50, 190, 120), source_page=0),
                    Cell("B", 0, 1, box=Box(100, 50, 190, 120), source_page=0),
                ],
            ),
        ])])
        codes = {issue.code for issue in broken.integrity_issues()}
        self.assertIn("paragraph_run_text_mismatch", codes)
        self.assertIn("run_provenance_mismatch", codes)
        self.assertIn("overlapping_table_cells", codes)

    def test_understood_page_has_no_model_integrity_violations(self):
        table = Table(Box(40, 140, 520, 210), 0, 1, 2, [
            Cell("Left", 0, 0, box=Box(40, 140, 280, 210)),
            Cell("Right", 0, 1, box=Box(280, 140, 520, 210)),
        ])
        words = [
            line("Body text continues across", 80, end=500, number=0),
            line("another visual line.", 96, end=350, number=1),
            line("Left", 160, x=60, end=100, number=2),
            line("Right", 160, x=320, end=370, number=3),
        ]
        page = understand_page(RecognizedPage(0, 595, 842, words, tables=[table]))
        self.assertEqual(Document([page]).integrity_issues(), [])

    def test_checked_in_model_examples_are_valid_schema_v2(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "model"
        examples = sorted(root.glob("*.json"))
        self.assertGreaterEqual(len(examples), 2)
        for path in examples:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["schema_version"], 2, path.name)
            self.assertTrue(value["pages"], path.name)
            for page in value["pages"]:
                self.assertGreater(page["width"], 0)
                self.assertGreater(page["height"], 0)

    def test_language_profiles_require_no_caller_language(self):
        self.assertEqual(script_counts("Hello Привет 中文 العربية").keys(), {"Latin", "Cyrillic", "Han", "Arabic"})
        self.assertEqual(profile("Cyrillic", {"eng", "rus", "osd"}), "rus+eng")
        self.assertEqual(profile("Mixed", {"eng", "rus", "ara", "chi_sim", "osd"}), "eng+rus+ara+chi_sim")
        with self.assertRaises(EngineError):
            profile("Arabic", {"eng", "osd"})


class ReconstructionTests(unittest.TestCase):
    def test_rtl_cjk_tables_and_no_hard_line_breaks(self):
        paragraphs = [Paragraph("مرحبا بالعالم 123 OpenAI", Box(48, 48, 520, 80), 0, direction="rtl"), Paragraph("中文内容 English 内容", Box(48, 90, 520, 120), 0)]
        table = Table(Box(48, 130, 520, 180), 0, 2, 2, [Cell("Merged header", 0, 0, column_span=2), Cell("مرحبا", 1, 0, direction="rtl"), Cell("中文", 1, 1)])
        model = Document([Page(0, 595, 842, [*paragraphs, table])])
        output = reconstruct(model)
        report = verify(output, model)
        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.tables, 1)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").decode()
            self.assertIn("Noto Sans CJK SC", xml)
            self.assertIn("Noto Naskh Arabic", xml)
            self.assertIn("gridSpan", xml)
            self.assertIn("مرحبا", xml)


    def test_inline_style_hints_are_reconstructed_without_recognition_imports(self):
        runs = [
            InlineRun("Bold", Box(48, 48, 82, 62), 0, bold=True, size=12, source_region="native_pdf"),
            InlineRun(" italic", Box(84, 48, 130, 62), 0, italic=True, size=11, source_region="native_pdf"),
            InlineRun(" العربية", Box(132, 48, 205, 62), 0, direction="rtl", italic=True, size=11, script="Arabic", source_region="native_pdf"),
        ]
        paragraph = Paragraph("Bold italic العربية", Box(48, 48, 205, 62), 0, runs=runs)
        output = reconstruct(Document([Page(0, 595, 842, [paragraph])]))
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").decode()
        self.assertIn("<w:b/>", xml)
        self.assertIn("<w:i/>", xml)
        self.assertIn("w:rtl", xml)
        self.assertIn("Noto Naskh Arabic", xml)
        self.assertNotIn("document_reconstruction.recognition", xml)


    def test_heading_levels_and_ordered_unordered_lists_reconstruct_editably(self):
        model = Document([Page(0, 595, 842, [
            Paragraph("Scope", Box(48, 48, 300, 70), 0, kind="heading", level=1),
            Paragraph("• First bullet", Box(60, 80, 300, 96), 0, kind="list_item", list_marker="•", list_ordered=False),
            Paragraph("a. First lettered item", Box(60, 100, 340, 116), 0, kind="list_item", list_marker="a.", list_ordered=True),
            Paragraph("b. Second lettered item", Box(60, 120, 340, 136), 0, kind="list_item", list_marker="b.", list_ordered=True),
        ])])
        output = reconstruct(model)
        report = verify(output, model)
        self.assertTrue(report.passed, report.failures)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").decode()
            numbering = package.read("word/numbering.xml").decode()
        self.assertIn('w:val="Subheading"', xml)
        self.assertIn('w:val="lowerLetter"', numbering)
        self.assertNotIn("• First bullet", xml)
        self.assertIn("First bullet", xml)

    def test_table_cell_inline_runs_and_nested_list_levels_reconstruct_editably(self):
        cell_runs = [
            InlineRun("Bold", Box(48, 120, 80, 134), 0, bold=True, size=11, source_region="native_pdf"),
            InlineRun(" value", Box(82, 120, 125, 134), 0, italic=True, size=11, source_region="native_pdf"),
        ]
        model = Document([Page(0, 595, 842, [
            Paragraph("1. Parent", Box(48, 48, 220, 64), 0, kind="list_item", list_marker="1.", list_ordered=True, list_id=1, level=0),
            Paragraph("a. Child", Box(72, 68, 220, 84), 0, kind="list_item", list_marker="a.", list_ordered=True, list_id=1, level=1),
            Paragraph("2. Parent again", Box(48, 88, 240, 104), 0, kind="list_item", list_marker="2.", list_ordered=True, list_id=1, level=0),
            Table(Box(48, 120, 300, 160), 0, 1, 1, [Cell("Bold value", 0, 0, box=Box(48, 120, 300, 160), source_page=0, runs=cell_runs)]),
        ])])
        output = reconstruct(model)
        report = verify(output, model)
        self.assertTrue(report.passed, report.failures)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").decode()
            numbering_xml = package.read("word/numbering.xml").decode()
        self.assertIn("<w:b/>", xml)
        self.assertIn("<w:i/>", xml)
        self.assertIn('w:ilvl w:val="1"', xml)
        self.assertIn('<w:lvl w:ilvl="1">', numbering_xml)
        root = ET.fromstring(xml)
        numbered = []
        for paragraph in root.findall(".//w:body/w:p", NS):
            num_pr = paragraph.find("w:pPr/w:numPr", NS)
            if num_pr is None:
                continue
            level = num_pr.find("w:ilvl", NS).get(f"{{{NS['w']}}}val")
            number = num_pr.find("w:numId", NS).get(f"{{{NS['w']}}}val")
            numbered.append((level, number))
        self.assertEqual([level for level, _ in numbered[:3]], ["0", "1", "0"])
        self.assertEqual(numbered[0][1], numbered[2][1])
        self.assertNotEqual(numbered[0][1], numbered[1][1])

    def test_model_serialization_contains_structure_but_not_graphic_bytes(self):
        image = io.BytesIO()
        Image.new("RGB", (5, 5), "white").save(image, format="PNG")
        model = Document([Page(0, 595, 842, [
            Paragraph("Body", Box(48, 48, 120, 64), 0, runs=[InlineRun("Body", Box(48, 48, 120, 64), 0, source_region="native_pdf")]),
            Graphic(Box(48, 80, 90, 120), 0, image.getvalue(), role="logo"),
        ], source_rotation=90)], metadata={"example": "yes"}, issues=[StructuralIssue("example_issue", "Example unresolved issue.", "info", 0.5)])
        value = model.to_dict()
        self.assertEqual(value["schema_version"], 2)
        self.assertEqual(value["metadata"]["example"], "yes")
        self.assertEqual(value["pages"][0]["source_rotation"], 90)
        self.assertEqual(value["pages"][0]["blocks"][1]["data_bytes"], len(image.getvalue()))
        self.assertNotIn("data", value["pages"][0]["blocks"][1])

    def test_full_page_raster_is_rejected(self):
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), "white").save(buffer, format="PNG")
        model = Document([Page(0, 595, 842, [Graphic(Box(0, 0, 595, 842), 0, buffer.getvalue())])])
        with self.assertRaises(ValueError):
            reconstruct(model)

    def test_corrupt_package_fails(self):
        self.assertFalse(verify(b"not a zip", Document([])).passed)

    def test_missing_and_duplicated_text_fail(self):
        model = Document([Page(0, 595, 842, [Paragraph("Expected content", Box(48, 48, 520, 80), 0)])])
        output = reconstruct(model)
        changed = Document([Page(0, 595, 842, [Paragraph("Different content", Box(48, 48, 520, 80), 0)])])
        self.assertFalse(verify(output, changed).passed)

    def test_blank_source_page_is_explicit(self):
        model = Document([Page(0, 595, 842, [], intentionally_blank=True), Page(1, 595, 842, [Paragraph("A second page", Box(48, 48, 520, 80), 1)])])
        report = verify(reconstruct(model), model)
        self.assertTrue(report.passed, report.failures)
        self.assertEqual(report.logical_pages, 2)


class Stage6DocxHardeningTests(unittest.TestCase):
    @staticmethod
    def _rewrite_package(data: bytes, replacements: dict[str, bytes]) -> bytes:
        source = io.BytesIO(data)
        target = io.BytesIO()
        with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                value = replacements.get(item.filename, zin.read(item.filename))
                zout.writestr(item, value)
        return target.getvalue()

    def test_unordered_list_uses_numbering_and_preserves_inline_runs_after_marker(self):
        runs = [
            InlineRun("• ", Box(48, 48, 58, 62), 0),
            InlineRun("Bold", Box(60, 48, 92, 62), 0, bold=True),
            InlineRun(" tail", Box(94, 48, 130, 62), 0, italic=True),
        ]
        block = Paragraph("• Bold tail", Box(48, 48, 130, 62), 0, kind="list_item", list_marker="•", list_ordered=False, list_id=7, runs=runs)
        output = reconstruct(Document([Page(0, 595, 842, [block])]))
        self.assertTrue(verify(output, Document([Page(0, 595, 842, [block])])).passed)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            document_xml = package.read("word/document.xml").decode()
            numbering_xml = package.read("word/numbering.xml").decode()
        self.assertIn("<w:numPr>", document_xml)
        self.assertIn('w:val="bullet"', numbering_xml)
        self.assertIn("<w:b/>", document_xml)
        self.assertIn("<w:i/>", document_xml)
        self.assertNotIn("• Bold tail", document_xml)

    def test_table_does_not_invent_header_fixed_height_or_spacer(self):
        table = Table(Box(48, 80, 500, 160), 0, 2, 2, [
            Cell("A", 0, 0), Cell("B", 0, 1), Cell("C", 1, 0), Cell("D", 1, 1),
        ])
        output = reconstruct(Document([Page(0, 595, 842, [table])]))
        report = verify(output, Document([Page(0, 595, 842, [table])]))
        self.assertTrue(report.passed, report.failures)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            root = ET.fromstring(package.read("word/document.xml"))
        self.assertFalse(root.findall(".//w:tblHeader", NS))
        self.assertFalse(root.findall(".//w:trHeight", NS))
        body = root.find("w:body", NS)
        tags = [child.tag.rsplit("}", 1)[-1] for child in list(body)]
        self.assertEqual(tags[:-1], ["tbl"])

    def test_landscape_page_has_explicit_orientation(self):
        model = Document([Page(0, 842, 595, [Paragraph("Landscape", Box(48, 48, 200, 65), 0)])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").decode()
        self.assertIn('w:orient="landscape"', xml)

    def test_tall_graphic_is_fitted_without_aspect_distortion(self):
        source = io.BytesIO()
        Image.new("RGB", (100, 400), "white").save(source, format="PNG")
        model = Document([Page(0, 595, 842, [Graphic(Box(48, 80, 248, 280), 0, source.getvalue())])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            root = ET.fromstring(package.read("word/document.xml"))
        ns = {"wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}
        extent = root.find(".//wp:extent", ns)
        cx, cy = int(extent.get("cx")), int(extent.get("cy"))
        self.assertAlmostEqual(cy / cx, 4.0, delta=0.02)


    def test_quality_preserves_cjk_run_adjacency_without_invented_spaces(self):
        paragraph = Paragraph(
            "文件編號 2026-A17，金額123.45元；審核完成。",
            Box(48, 48, 500, 80),
            0,
            runs=[
                InlineRun("文件編號", Box(48, 48, 120, 64), 0, script="Han"),
                InlineRun(" 2026-A17，", Box(120, 48, 210, 64), 0, script="Latin"),
                InlineRun("金額", Box(210, 48, 250, 64), 0, script="Han"),
                InlineRun("123.45", Box(250, 48, 310, 64), 0, script="Latin"),
                InlineRun("元；審核完成。", Box(310, 48, 500, 64), 0, script="Han"),
            ],
        )
        model = Document([Page(0, 595, 842, [paragraph])])
        output = reconstruct(model)
        report = verify(output, model)
        self.assertTrue(report.passed, report.failures)

    def test_quality_rejects_broken_relationship_and_malformed_secondary_xml(self):
        model = Document([Page(0, 595, 842, [Paragraph("Body", Box(48, 48, 120, 64), 0)])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            rels = package.read("_rels/.rels").replace(b"word/document.xml", b"word/missing.xml")
        broken = self._rewrite_package(output, {"_rels/.rels": rels})
        self.assertIn("broken_relationship:_rels/.rels:word/missing.xml", verify(broken, model).failures)
        malformed = self._rewrite_package(output, {"word/styles.xml": b"<broken"})
        self.assertIn("invalid_xml:word/styles.xml", verify(malformed, model).failures)

    def test_quality_rejects_reordered_text_even_when_characters_match(self):
        model = Document([Page(0, 595, 842, [Paragraph("ABC DEF", Box(48, 48, 160, 64), 0)])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            xml = package.read("word/document.xml").replace(b"ABC DEF", b"DEF ABC")
        changed = self._rewrite_package(output, {"word/document.xml": xml})
        self.assertIn("text_missing_duplicated_or_reordered", verify(changed, model).failures)

    def test_quality_rejects_invalid_embedded_image(self):
        source = io.BytesIO()
        Image.new("RGB", (20, 10), "white").save(source, format="PNG")
        model = Document([Page(0, 595, 842, [Graphic(Box(48, 80, 148, 130), 0, source.getvalue())])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            media = next(name for name in package.namelist() if name.startswith("word/media/"))
        changed = self._rewrite_package(output, {media: b"not-an-image"})
        self.assertTrue(any(value.startswith("invalid_embedded_image:") for value in verify(changed, model).failures))

    def test_quality_rejects_unregistered_or_mislabelled_image_content_type(self):
        source = io.BytesIO()
        Image.new("RGBA", (20, 10), "red").save(source, format="PNG")
        model = Document([Page(0, 595, 842, [Graphic(Box(48, 80, 148, 130), 0, source.getvalue())])])
        output = reconstruct(model)
        self.assertTrue(verify(output, model).passed)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            content_types = ET.fromstring(package.read("[Content_Types].xml"))
        default = next(element for element in content_types if element.get("Extension") == "png")
        content_types.remove(default)
        missing_type = self._rewrite_package(output, {"[Content_Types].xml": ET.tostring(content_types)})
        self.assertIn("missing_image_content_type:word/media/image1.png", verify(missing_type, model).failures)

        default.set("ContentType", "image/jpeg")
        content_types.append(default)
        wrong_type = self._rewrite_package(output, {"[Content_Types].xml": ET.tostring(content_types)})
        self.assertIn("invalid_image_content_type:word/media/image1.png", verify(wrong_type, model).failures)

    def test_quality_rejects_missing_main_document_content_type(self):
        model = Document([Page(0, 595, 842, [Paragraph("Body", Box(48, 48, 120, 64), 0)])])
        output = reconstruct(model)
        with zipfile.ZipFile(io.BytesIO(output)) as package:
            content_types = ET.fromstring(package.read("[Content_Types].xml"))
        document_override = next(element for element in content_types if element.get("PartName") == "/word/document.xml")
        content_types.remove(document_override)
        changed = self._rewrite_package(output, {"[Content_Types].xml": ET.tostring(content_types)})
        self.assertIn("invalid_main_document_content_type", verify(changed, model).failures)

    def test_quality_rejects_duplicate_zip_members(self):
        model = Document([Page(0, 595, 842, [Paragraph("Body", Box(48, 48, 120, 64), 0)])])
        output = reconstruct(model)
        target = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(output)) as original, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as rewritten:
            for item in original.infolist():
                rewritten.writestr(item, original.read(item.filename))
            # The second member shadows the first for name-based reads.
            rewritten.writestr("word/document.xml", original.read("word/document.xml"))
        self.assertIn("duplicate_zip_member:word/document.xml", verify(target.getvalue(), model).failures)


if __name__ == "__main__":
    unittest.main()
