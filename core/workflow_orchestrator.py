"""
Context-Aware Multi-Step Workflow Orchestrator
Executes multi-step document intelligence pipelines with state persistence,
intermediate artifact chaining, progress tracking, and validation/evaluation hooks.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from .ingestion import DocumentIngestor, IngestedDocument
from .prompt_engine import PromptEngine
from .llm_client import LLMClient
from .validator import OutputValidator, ValidationResult
from .evaluation import EvaluationEngine, EvaluationReport


# =====================================================================
# Workflow Context & State
# =====================================================================

@dataclass
class StepRecord:
    step_id: str
    name: str
    description: str
    status: str  # "PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"
    duration_ms: float = 0.0
    tokens_used: int = 0
    output_key: Optional[str] = None
    output_preview: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_used": self.tokens_used,
            "output_key": self.output_key,
            "output_preview": self.output_preview[:200] + ("..." if len(self.output_preview) > 200 else ""),
            "error_message": self.error_message,
        }


@dataclass
class WorkflowContext:
    workflow_id: str
    document: IngestedDocument
    user_requirements: str
    target_audience: str = "Executive Stakeholders"
    intermediate_artifacts: Dict[str, Any] = field(default_factory=dict)
    step_history: List[StepRecord] = field(default_factory=list)
    final_output: str = ""
    validation_result: Optional[ValidationResult] = None
    evaluation_report: Optional[EvaluationReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "document_id": self.document.doc_id,
            "filename": self.document.filename,
            "user_requirements": self.user_requirements,
            "target_audience": self.target_audience,
            "intermediate_artifacts_keys": list(self.intermediate_artifacts.keys()),
            "step_history": [s.to_dict() for s in self.step_history],
            "final_output_preview": self.final_output[:300] + ("..." if len(self.final_output) > 300 else ""),
            "is_valid": self.validation_result.is_valid if self.validation_result else None,
            "quality_index": self.evaluation_report.overall_quality_index if self.evaluation_report else None,
            "metadata": self.metadata,
        }


# =====================================================================
# Workflow Step Definition
# =====================================================================

@dataclass
class WorkflowStep:
    step_id: str
    name: str
    description: str
    action_type: str  # "ingest", "classify", "extract", "generate", "validate", "evaluate"
    template_id: Optional[str] = None
    output_key: str = "step_output"
    parameters: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Workflow Pipeline Definition
# =====================================================================

class WorkflowPipeline:
    def __init__(self, pipeline_id: str, name: str, description: str, steps: List[WorkflowStep]):
        self.pipeline_id = pipeline_id
        self.name = name
        self.description = description
        self.steps = steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "steps_count": len(self.steps),
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "description": s.description,
                    "action_type": s.action_type,
                    "output_key": s.output_key,
                }
                for s in self.steps
            ]
        }


# =====================================================================
# Main Orchestrator Engine
# =====================================================================

class WorkflowOrchestrator:
    """Coordinates and executes multi-step context-aware workflows."""

    def __init__(
        self,
        ingestor: Optional[DocumentIngestor] = None,
        prompt_engine: Optional[PromptEngine] = None,
        llm_client: Optional[LLMClient] = None,
        validator: Optional[OutputValidator] = None,
        evaluation_engine: Optional[EvaluationEngine] = None,
    ):
        self.ingestor = ingestor or DocumentIngestor()
        self.prompt_engine = prompt_engine or PromptEngine()
        self.llm_client = llm_client or LLMClient()
        self.validator = validator or OutputValidator()
        self.evaluation_engine = evaluation_engine or EvaluationEngine(llm_client=self.llm_client)
        self._pipelines: Dict[str, WorkflowPipeline] = {}
        self._register_default_pipelines()

    def _register_default_pipelines(self):
        # 1. End-to-End Enterprise Document Intelligence Pipeline
        self.register_pipeline(WorkflowPipeline(
            pipeline_id="full_doc_intel",
            name="End-to-End Document Intelligence & Quality Pipeline",
            description="Complete multi-step pipeline: Document Classification -> Entity & KPI Extraction -> Context-Aware Executive Synthesis -> Output Validation -> Multi-Criteria Evaluation.",
            steps=[
                WorkflowStep(
                    step_id="step_1_classify",
                    name="Document Classification & Risk Profiling",
                    description="Identifies domain, document sub-type, urgency, confidentiality, and compliance exposures.",
                    action_type="classify",
                    template_id="cls_taxonomy_risk",
                    output_key="classification_profile"
                ),
                WorkflowStep(
                    step_id="step_2_extract",
                    name="Structured Entity & Metric Extraction",
                    description="Extracts key organizations, quantitative metrics, dates, and contractual constraints into structured tables.",
                    action_type="extract",
                    template_id="ext_entities_metrics",
                    output_key="extracted_entities"
                ),
                WorkflowStep(
                    step_id="step_3_synthesize",
                    name="Context-Aware Executive Synthesis",
                    description="Synthesizes findings, leveraging entities and classifications from previous steps into a comprehensive briefing.",
                    action_type="generate",
                    template_id="sum_exec_brief",
                    output_key="executive_briefing"
                ),
                WorkflowStep(
                    step_id="step_4_validate",
                    name="Quality Assurance & Grounding Validation",
                    description="Runs Pydantic/JSON validation, checks factual grounding against source text, and scans for PII.",
                    action_type="validate",
                    output_key="validation_result"
                ),
                WorkflowStep(
                    step_id="step_5_evaluate",
                    name="Multi-Dimensional Output Evaluation",
                    description="Scores Relevance, Consistency, Factuality, Completeness, Readability, and ROUGE metrics.",
                    action_type="evaluate",
                    output_key="evaluation_report"
                ),
            ]
        ))

        # 2. Strategic Action Memo & Execution Roadmap Pipeline
        self.register_pipeline(WorkflowPipeline(
            pipeline_id="strategic_action_roadmap",
            name="Strategic Action Memo & Phased Roadmap",
            description="Analyzes requirements and document to build a phased 30-60-90 day strategic execution roadmap with quality checks.",
            steps=[
                WorkflowStep(
                    step_id="step_1_extract",
                    name="Core Parameter Extraction",
                    description="Extracts operational constraints, metrics, and milestones.",
                    action_type="extract",
                    template_id="ext_entities_metrics",
                    output_key="extracted_parameters"
                ),
                WorkflowStep(
                    step_id="step_2_roadmap",
                    name="Action Memo & Roadmap Generation",
                    description="Generates phased implementation memo with immediate triage, execution scaling, and governance milestones.",
                    action_type="generate",
                    template_id="gen_action_plan",
                    output_key="action_roadmap"
                ),
                WorkflowStep(
                    step_id="step_3_validate",
                    name="Factual Grounding & Completeness Verification",
                    description="Verifies grounding of dates, metrics, and deliverables against source text.",
                    action_type="validate",
                    output_key="validation_result"
                ),
                WorkflowStep(
                    step_id="step_4_evaluate",
                    name="Relevance & Completeness Evaluation",
                    description="Evaluates user intent alignment, readability, and structural completeness.",
                    action_type="evaluate",
                    output_key="evaluation_report"
                ),
            ]
        ))

        # 3. Knowledge Base & High-Yield FAQ Pipeline
        self.register_pipeline(WorkflowPipeline(
            pipeline_id="faq_knowledge_base",
            name="Knowledge Base & FAQ Extraction Pipeline",
            description="Automatically turns technical, clinical, or financial documents into a verifiable Q&A knowledge base.",
            steps=[
                WorkflowStep(
                    step_id="step_1_classify",
                    name="Domain & Topic Classification",
                    description="Maps subject matter and identifies high-priority reader concerns.",
                    action_type="classify",
                    template_id="cls_taxonomy_risk",
                    output_key="domain_context"
                ),
                WorkflowStep(
                    step_id="step_2_faq_gen",
                    name="FAQ & Knowledge Generation",
                    description="Generates 5-8 verified question-answer pairs with source citations.",
                    action_type="generate",
                    template_id="gen_faq_knowledge",
                    output_key="faq_knowledge"
                ),
                WorkflowStep(
                    step_id="step_3_evaluate",
                    name="Factuality & Readability Benchmark",
                    description="Checks answer factuality against source citations and assesses Flesch readability.",
                    action_type="evaluate",
                    output_key="evaluation_report"
                ),
            ]
        ))

    def register_pipeline(self, pipeline: WorkflowPipeline):
        self._pipelines[pipeline.pipeline_id] = pipeline

    def list_pipelines(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._pipelines.values()]

    def get_pipeline(self, pipeline_id: str) -> Optional[WorkflowPipeline]:
        return self._pipelines.get(pipeline_id)

    def execute_pipeline(
        self,
        pipeline_id: str,
        document: IngestedDocument,
        user_requirements: str = "",
        target_audience: str = "Stakeholders",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> WorkflowContext:
        """
        Executes a multi-step pipeline sequentially, passing context and accumulating intermediate artifacts.
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise KeyError(f"Pipeline '{pipeline_id}' not found.")

        context = WorkflowContext(
            workflow_id=f"wf_{int(time.time()*1000)}",
            document=document,
            user_requirements=user_requirements or "Synthesize primary findings and actionable intelligence.",
            target_audience=target_audience,
            metadata={"pipeline_id": pipeline_id, "pipeline_name": pipeline.name}
        )

        for step in pipeline.steps:
            record = StepRecord(
                step_id=step.step_id,
                name=step.name,
                description=step.description,
                status="RUNNING",
                output_key=step.output_key
            )
            context.step_history.append(record)

            if progress_callback:
                progress_callback({"status": "running", "step": record.to_dict()})

            step_start = time.time()
            try:
                self._execute_single_step(step, context, record)
                record.status = "COMPLETED"
                record.duration_ms = (time.time() - step_start) * 1000
                if progress_callback:
                    progress_callback({"status": "completed", "step": record.to_dict()})
            except Exception as e:
                record.status = "FAILED"
                record.duration_ms = (time.time() - step_start) * 1000
                record.error_message = str(e)
                if progress_callback:
                    progress_callback({"status": "failed", "step": record.to_dict(), "error": str(e)})
                break

        return context

    def _execute_single_step(self, step: WorkflowStep, context: WorkflowContext, record: StepRecord):
        """Dispatches step execution based on action_type."""
        doc_text = context.document.clean_text

        # 1. Classification Step
        if step.action_type == "classify":
            template_id = step.template_id or "cls_taxonomy_risk"
            rendered = self.prompt_engine.render(template_id, {
                "requirements": context.user_requirements,
                "document_text": doc_text[:4000]
            })
            resp = self.llm_client.generate(
                prompt=rendered["user_prompt"],
                system_prompt=rendered["system_prompt"],
                temperature=rendered["temperature"],
                category="classification"
            )
            context.intermediate_artifacts[step.output_key] = resp.content
            record.tokens_used = resp.total_tokens
            record.output_preview = resp.content[:200]

        # 2. Extraction Step
        elif step.action_type == "extract":
            template_id = step.template_id or "ext_entities_metrics"
            rendered = self.prompt_engine.render(template_id, {
                "requirements": context.user_requirements,
                "document_text": doc_text[:4000]
            })
            resp = self.llm_client.generate(
                prompt=rendered["user_prompt"],
                system_prompt=rendered["system_prompt"],
                temperature=rendered["temperature"],
                category="extraction"
            )
            context.intermediate_artifacts[step.output_key] = resp.content
            record.tokens_used = resp.total_tokens
            record.output_preview = resp.content[:200]

        # 3. Content Generation / Synthesis Step (Context-Aware!)
        elif step.action_type == "generate":
            template_id = step.template_id or "sum_exec_brief"
            # Enrich requirements with context from previous steps if available!
            enriched_req = context.user_requirements
            if "classification_profile" in context.intermediate_artifacts:
                enriched_req += f"\n(Incorporate context from classification: {context.intermediate_artifacts['classification_profile'][:150]}...)"
            if "extracted_entities" in context.intermediate_artifacts:
                enriched_req += f"\n(Incorporate extracted metrics: {context.intermediate_artifacts['extracted_entities'][:200]}...)"

            rendered = self.prompt_engine.render(template_id, {
                "requirements": enriched_req,
                "target_audience": context.target_audience,
                "document_text": doc_text[:4500]
            })
            resp = self.llm_client.generate(
                prompt=rendered["user_prompt"],
                system_prompt=rendered["system_prompt"],
                temperature=rendered["temperature"],
                category="content_generation"
            )
            context.intermediate_artifacts[step.output_key] = resp.content
            context.final_output = resp.content
            record.tokens_used = resp.total_tokens
            record.output_preview = resp.content[:200]

        # 4. Validation Step
        elif step.action_type == "validate":
            target_text = context.final_output or list(context.intermediate_artifacts.values())[-1]
            val_res = self.validator.validate_output(
                generated_output=target_text,
                source_text=doc_text,
                expected_format="markdown",
                min_words=15
            )
            context.validation_result = val_res
            context.intermediate_artifacts[step.output_key] = val_res.to_dict()
            record.output_preview = f"Quality Score: {val_res.quality_score}% | Valid: {val_res.is_valid} | Errors: {len(val_res.errors)}"

        # 5. Evaluation Step
        elif step.action_type == "evaluate":
            target_text = context.final_output or list(context.intermediate_artifacts.values())[-1]
            eval_rep = self.evaluation_engine.evaluate(
                generated_text=target_text,
                source_text=doc_text,
                requirements=context.user_requirements,
                run_llm_judge=False
            )
            context.evaluation_report = eval_rep
            context.intermediate_artifacts[step.output_key] = eval_rep.to_dict()
            record.output_preview = f"Index: {round(eval_rep.overall_quality_index, 1)}% ({eval_rep.grade}) | Factuality: {eval_rep.factuality.rating}"

        else:
            raise ValueError(f"Unknown action type '{step.action_type}'")
