"""
Unit and Integration Test Suite
Validates Ingestion, Prompt Engine, LLM Client, Validator, Evaluation, and Multi-Step Orchestration.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ingestion import DocumentIngestor, IngestedDocument
from core.prompt_engine import PromptEngine, PromptTemplate, TaskCategory
from core.llm_client import LLMClient, LocalIntelligentEngine
from core.validator import OutputValidator, DocumentIntelligenceModel
from core.evaluation import EvaluationEngine
from core.workflow_orchestrator import WorkflowOrchestrator


class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.ingestor = DocumentIngestor()
        self.sample_text = (
            "# APEX TECHNOLOGIES INC.\n\n"
            "Total consolidated revenue in fiscal year 2025 reached $4,850,000,000, "
            "up 22.4% year-over-year. Cloud gross margins reached 71.2%.\n\n"
            "SECTION 2: CASH FLOW\n"
            "Operating cash flow reached $940,000,000 with strong liquidity reserves."
        )

    def test_text_normalization(self):
        dirty = "Hello   world \u201csmart quotes\u201d\r\n\r\n\r\nNext line."
        cleaned = self.ingestor.preprocess_text(dirty)
        self.assertIn('"smart quotes"', cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_statistics(self):
        stats = self.ingestor.calculate_statistics(self.sample_text)
        self.assertGreater(stats["word_count"], 20)
        self.assertGreater(stats["character_count"], 100)
        self.assertIn("lexical_diversity", stats)
        self.assertTrue(len(stats["top_keywords"]) > 0)

    def test_chunking_strategies(self):
        # Paragraph chunking
        p_chunks = self.ingestor.chunk_text(self.sample_text, strategy="paragraph", chunk_size=50)
        self.assertGreaterEqual(len(p_chunks), 1)

        # Fixed window chunking
        w_chunks = self.ingestor.chunk_text(self.sample_text, strategy="fixed_window", chunk_size=15, chunk_overlap=5)
        self.assertGreaterEqual(len(w_chunks), 2)

        # Section chunking
        s_chunks = self.ingestor.chunk_text(self.sample_text, strategy="section")
        self.assertGreaterEqual(len(s_chunks), 1)

    def test_sample_files_exist_and_ingest(self):
        sample_file = BASE_DIR / "data" / "samples" / "annual_financial_report.txt"
        self.assertTrue(sample_file.exists())
        doc = self.ingestor.ingest_file(str(sample_file))
        self.assertEqual(doc.file_type, "txt")
        self.assertGreater(len(doc.clean_text), 500)


class TestPromptEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PromptEngine()

    def test_default_templates_registered(self):
        templates = self.engine.list_templates()
        self.assertGreaterEqual(len(templates), 6)
        ids = [t["template_id"] for t in templates]
        self.assertIn("sum_exec_brief", ids)
        self.assertIn("ext_entities_metrics", ids)
        self.assertIn("cls_taxonomy_risk", ids)
        self.assertIn("gen_action_plan", ids)

    def test_render_with_variables(self):
        rendered = self.engine.render("sum_exec_brief", {
            "requirements": "Focus on cash flow and cloud margins",
            "document_text": "Sample text for briefing."
        })
        self.assertIn("Focus on cash flow and cloud margins", rendered["user_prompt"])
        self.assertIn("Sample text for briefing.", rendered["user_prompt"])
        self.assertIn("system_prompt", rendered)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = OutputValidator()
        self.source_text = "Apex Dynamics reported $4,850,000,000 revenue with 22.4% growth."

    def test_json_repair(self):
        malformed = '```json\n{"title": "Test", "val": 100,}\n```'
        parsed, was_repaired = self.validator.repair_json_string(malformed)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "Test")

    def test_grounding_check(self):
        grounded_output = "The total revenue was $4,850,000,000 with 22.4% growth rate."
        res = self.validator.check_grounding(grounded_output, self.source_text)
        self.assertGreaterEqual(res.grounding_ratio, 0.8)

        hallucinated_output = "The total revenue was $9,999,999,999 with 88.8% growth."
        res_hall = self.validator.check_grounding(hallucinated_output, self.source_text)
        self.assertLess(res_hall.grounding_ratio, 0.5)

    def test_pydantic_schema_validation(self):
        valid_json = (
            '{\n'
            '  "title": "Quarterly Report",\n'
            '  "document_type": "Financial",\n'
            '  "summary": "This is a detailed narrative summary meeting minimum length.",\n'
            '  "key_findings": ["Finding A", "Finding B"],\n'
            '  "entities": [{"name": "Apex", "category": "ORG"}],\n'
            '  "metrics": [{"label": "Revenue", "value": "$4.8B", "unit": "USD"}],\n'
            '  "risk_assessment": {"overall_risk": "LOW", "risk_factors": ["None"]},\n'
            '  "action_items": [{"task": "Review data", "priority": "HIGH"}]\n'
            '}'
        )
        res = self.validator.validate_output(valid_json, source_text="Quarterly Report", expected_format="json", schema_name="document_intelligence")
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluator = EvaluationEngine()
        self.source = "Apex Dynamics reported record revenue of $4.85B with 22.4% growth."
        self.generated = "Apex Dynamics announced revenue of $4.85B reflecting 22.4% expansion."

    def test_rouge_metrics(self):
        rouge = self.evaluator.compute_rouge(self.generated, self.source)
        self.assertGreater(rouge.rouge_1_f1, 0.5)
        self.assertGreater(rouge.rouge_l_f1, 0.4)

    def test_readability(self):
        simple_text = "The company makes good software. It runs fast. Users are happy."
        score = self.evaluator.evaluate_readability(simple_text)
        self.assertGreater(score.percentage, 50.0)

    def test_end_to_end_evaluation(self):
        report = self.evaluator.evaluate(self.generated, source_text=self.source, requirements="Report revenue figures.")
        self.assertGreater(report.overall_quality_index, 60.0)
        self.assertIn(report.grade, ["A+", "A", "B", "C"])
        self.assertIn("Factuality", report.radar_data)
        self.assertIn("Relevance", report.radar_data)


class TestWorkflowOrchestrator(unittest.TestCase):
    def setUp(self):
        self.ingestor = DocumentIngestor()
        # Use local mode for rapid deterministic unit tests
        self.llm = LLMClient(mode="local")
        self.orchestrator = WorkflowOrchestrator(llm_client=self.llm)

    def test_full_pipeline_execution(self):
        sample_path = BASE_DIR / "data" / "samples" / "annual_financial_report.txt"
        doc = self.ingestor.ingest_file(str(sample_path))

        context = self.orchestrator.execute_pipeline(
            pipeline_id="full_doc_intel",
            document=doc,
            user_requirements="Synthesize cloud margins and earnings guidance."
        )

        self.assertIsNotNone(context.workflow_id)
        self.assertGreaterEqual(len(context.step_history), 5)
        for step in context.step_history:
            self.assertEqual(step.status, "COMPLETED")

        self.assertIn("classification_profile", context.intermediate_artifacts)
        self.assertIn("extracted_entities", context.intermediate_artifacts)
        self.assertIn("executive_briefing", context.intermediate_artifacts)
        self.assertIsNotNone(context.validation_result)
        self.assertIsNotNone(context.evaluation_report)


if __name__ == "__main__":
    unittest.main()
