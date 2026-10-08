"""
DocIntel & GenAI Platform - Core Package
AI-Powered Generative Content & Document Intelligence Platform
"""

from .ingestion import DocumentIngestor, DocumentChunk, IngestedDocument
from .prompt_engine import PromptEngine, PromptTemplate, TaskCategory
from .llm_client import LLMClient, ModelResponse
from .validator import OutputValidator, ValidationResult, ValidationErrorItem
from .evaluation import EvaluationEngine, EvaluationReport
from .workflow_orchestrator import WorkflowOrchestrator, WorkflowStep, WorkflowContext, WorkflowPipeline

__all__ = [
    "DocumentIngestor",
    "DocumentChunk",
    "IngestedDocument",
    "PromptEngine",
    "PromptTemplate",
    "TaskCategory",
    "LLMClient",
    "ModelResponse",
    "OutputValidator",
    "ValidationResult",
    "ValidationErrorItem",
    "EvaluationEngine",
    "EvaluationReport",
    "WorkflowOrchestrator",
    "WorkflowStep",
    "WorkflowContext",
    "WorkflowPipeline",
]
