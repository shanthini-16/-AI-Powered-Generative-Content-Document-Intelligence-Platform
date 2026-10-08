"""
Output Validation and Quality Assurance Guardrails
Implements Pydantic schema validation, JSON self-repair, hallucination/grounding checks,
completeness checks, repetition detection, and PII leakage scanning.
"""

import re
import json
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError


# =====================================================================
# Core Pydantic Schemas for Document Intelligence
# =====================================================================

class EntityItem(BaseModel):
    name: str = Field(..., description="Entity name")
    category: str = Field("GENERAL", description="Entity category: ORG, PERSON, LOCATION, TECH, etc.")
    relevance: Optional[str] = Field(None, description="Contextual relevance")


class MetricItem(BaseModel):
    label: str = Field(..., description="Metric label")
    value: str = Field(..., description="Metric value")
    unit: Optional[str] = Field("", description="Unit of measurement or currency")


class ActionItem(BaseModel):
    task: str = Field(..., description="Actionable task description")
    priority: str = Field("MEDIUM", description="HIGH, MEDIUM, or LOW")
    assigned_to: Optional[str] = Field("Unassigned", description="Assignee or stakeholder")


class RiskAssessmentModel(BaseModel):
    overall_risk: str = Field("MEDIUM", description="LOW, MEDIUM, HIGH, or CRITICAL")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")


class DocumentIntelligenceModel(BaseModel):
    title: str = Field(..., description="Document or briefing title")
    document_type: str = Field("General", description="Category or taxonomy of the document")
    summary: str = Field(..., min_length=15, description="High-level narrative summary")
    key_findings: List[str] = Field(default_factory=list, min_length=1, description="Bullet points of verified facts")
    entities: List[EntityItem] = Field(default_factory=list, description="Extracted entities")
    metrics: List[MetricItem] = Field(default_factory=list, description="Quantitative metrics")
    risk_assessment: RiskAssessmentModel = Field(default_factory=RiskAssessmentModel)
    action_items: List[ActionItem] = Field(default_factory=list)


# =====================================================================
# Validation Reporting Dataclasses
# =====================================================================

@dataclass
class ValidationErrorItem:
    code: str
    message: str
    severity: str  # "ERROR" or "WARNING"
    field: Optional[str] = None


@dataclass
class GroundingCheckResult:
    total_facts_checked: int
    verified_facts: int
    unverified_facts: int
    grounding_ratio: float  # 0.0 to 1.0
    verified_items: List[str] = field(default_factory=list)
    unverified_items: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    is_valid: bool
    quality_score: float  # 0 to 100
    errors: List[ValidationErrorItem]
    warnings: List[ValidationErrorItem]
    grounding: GroundingCheckResult
    parsed_json: Optional[Dict[str, Any]] = None
    was_repaired: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 1),
            "errors": [{"code": e.code, "message": e.message, "severity": e.severity, "field": e.field} for e in self.errors],
            "warnings": [{"code": w.code, "message": w.message, "severity": w.severity, "field": w.field} for w in self.warnings],
            "grounding": {
                "total_facts_checked": self.grounding.total_facts_checked,
                "verified_facts": self.grounding.verified_facts,
                "unverified_facts": self.grounding.unverified_facts,
                "grounding_ratio": round(self.grounding.grounding_ratio, 2),
                "verified_items": self.grounding.verified_items[:10],
                "unverified_items": self.grounding.unverified_items[:10],
            },
            "was_repaired": self.was_repaired,
            "parsed_json": self.parsed_json,
            "details": self.details,
        }


# =====================================================================
# Main OutputValidator Engine
# =====================================================================

class OutputValidator:
    """Validates generated outputs for schema conformance, factual grounding, safety, and formatting."""

    SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
        "document_intelligence": DocumentIntelligenceModel,
        "default": DocumentIntelligenceModel,
    }

    def __init__(self):
        pass

    def repair_json_string(self, text: str) -> tuple[Optional[Dict[str, Any]], bool]:
        """Attempts to parse and repair JSON with common LLM formatting idiosyncrasies."""
        cleaned = text.strip()

        # Remove markdown code fence ```json ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(cleaned), False
        except Exception:
            pass

        # Try repairing trailing commas before braces or brackets
        repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)

        # Try finding outer braces if preamble text exists
        start = repaired.find("{")
        end = repaired.rfind("}")
        if start != -1 and end != -1 and end > start:
            repaired = repaired[start:end + 1]

        try:
            return json.loads(repaired), True
        except Exception:
            return None, False

    def check_grounding(self, generated_text: str, source_text: str) -> GroundingCheckResult:
        """
        Cross-examines numbers, currency amounts, percentages, and uppercase named entities
        in the generated output against the source document.
        """
        if not source_text or not generated_text:
            return GroundingCheckResult(0, 0, 0, 1.0)

        source_lower = source_text.lower()

        # Extract specific factual markers: currency, percentages, numerical quantities
        num_patterns = [
            r"\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|k))?",  # Currencies
            r"\b\d+(?:\.\d+)?%",  # Percentages
            r"\b\d{4}\b",  # Years
        ]

        facts_to_check = []
        for pat in num_patterns:
            matches = re.findall(pat, generated_text, re.IGNORECASE)
            facts_to_check.extend([m.strip() for m in matches])

        # Also check capitalized proper noun entities (2+ words)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", generated_text)
        facts_to_check.extend(proper_nouns[:8])

        facts_unique = list(dict.fromkeys(facts_to_check))
        if not facts_unique:
            return GroundingCheckResult(0, 0, 0, 1.0)

        verified = []
        unverified = []

        for fact in facts_unique:
            fact_clean = fact.lower().strip()
            # Simple normalization check: strip dollar sign or commas
            fact_norm = re.sub(r"[$,]", "", fact_clean)
            if fact_clean in source_lower or fact_norm in source_lower:
                verified.append(fact)
            else:
                unverified.append(fact)

        total = len(facts_unique)
        v_count = len(verified)
        ratio = round(v_count / total, 3) if total > 0 else 1.0

        return GroundingCheckResult(
            total_facts_checked=total,
            verified_facts=v_count,
            unverified_facts=len(unverified),
            grounding_ratio=ratio,
            verified_items=verified,
            unverified_items=unverified,
        )

    def check_pii_and_safety(self, text: str) -> List[ValidationErrorItem]:
        """Detects sensitive leaks such as credit card numbers or raw SSNs."""
        warnings = []
        # SSN pattern: 000-00-0000
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
            warnings.append(ValidationErrorItem(
                code="PII_SSN_DETECTED",
                message="Potential Social Security Number (SSN) detected in generated text.",
                severity="WARNING"
            ))

        # Credit card pattern: 16 digits with dashes or spaces
        if re.search(r"\b(?:\d{4}[ -]?){3}\d{4}\b", text):
            warnings.append(ValidationErrorItem(
                code="PII_CREDIT_CARD_DETECTED",
                message="Potential credit card number sequence detected.",
                severity="WARNING"
            ))

        return warnings

    def check_repetition(self, text: str) -> List[ValidationErrorItem]:
        """Detects pathological token or sentence loops."""
        warnings = []
        sentences = [s.strip().lower() for s in re.split(r"[.!?]+(?:\s+|$)", text) if len(s.strip()) > 15]
        if len(sentences) >= 4:
            seen = set()
            duplicates = 0
            for s in sentences:
                if s in seen:
                    duplicates += 1
                seen.add(s)
            dup_ratio = duplicates / len(sentences)
            if dup_ratio > 0.25:
                warnings.append(ValidationErrorItem(
                    code="HIGH_REPETITION",
                    message=f"High sentence repetition detected ({round(dup_ratio*100, 1)}% duplicated).",
                    severity="WARNING"
                ))
        return warnings

    def validate_output(
        self,
        generated_output: str,
        source_text: str = "",
        expected_format: str = "markdown",
        schema_name: Optional[str] = None,
        min_words: int = 20,
    ) -> ValidationResult:
        """
        Runs comprehensive validation across syntax, schema, grounding, and quality guards.
        """
        errors: List[ValidationErrorItem] = []
        warnings: List[ValidationErrorItem] = []
        parsed_json = None
        was_repaired = False

        words = generated_output.split()
        word_count = len(words)

        # 1. Word count / Completeness check
        if word_count < min_words:
            errors.append(ValidationErrorItem(
                code="OUTPUT_TOO_SHORT",
                message=f"Output is too short ({word_count} words). Minimum required is {min_words} words.",
                severity="ERROR"
            ))

        # 2. Safety and repetition checks
        warnings.extend(self.check_pii_and_safety(generated_output))
        warnings.extend(self.check_repetition(generated_output))

        # 3. JSON & Pydantic Schema Validation (if applicable)
        if expected_format == "json" or schema_name:
            data, repaired = self.repair_json_string(generated_output)
            was_repaired = repaired
            if data is None:
                errors.append(ValidationErrorItem(
                    code="INVALID_JSON",
                    message="Generated response is not valid JSON and could not be repaired.",
                    severity="ERROR"
                ))
            else:
                parsed_json = data
                # Run Pydantic schema validation if schema recognized
                target_schema = self.SCHEMA_REGISTRY.get(schema_name or "default")
                if target_schema:
                    try:
                        target_schema.model_validate(data)
                    except ValidationError as ve:
                        for err in ve.errors():
                            loc = " -> ".join(str(l) for l in err.get("loc", []))
                            msg = err.get("msg", "Validation error")
                            errors.append(ValidationErrorItem(
                                code="PYDANTIC_SCHEMA_VIOLATION",
                                message=f"Field '{loc}': {msg}",
                                severity="ERROR",
                                field=loc
                            ))

        # 4. Factual Grounding Check
        grounding = self.check_grounding(generated_output, source_text)
        if grounding.unverified_facts > 0 and grounding.grounding_ratio < 0.6:
            warnings.append(ValidationErrorItem(
                code="POTENTIAL_HALLUCINATION",
                message=f"Low grounding ratio ({round(grounding.grounding_ratio*100, 1)}%). {grounding.unverified_facts} metrics/entities not found in source document.",
                severity="WARNING"
            ))

        # 5. Calculate Composite Quality Score (0 to 100)
        base_score = 100.0
        # Penalties:
        # Errors: -25 each
        # Warnings: -10 each
        # Grounding multiplier:
        base_score -= len(errors) * 25.0
        base_score -= len(warnings) * 10.0
        if grounding.total_facts_checked > 0:
            base_score = base_score * 0.7 + (grounding.grounding_ratio * 100.0) * 0.3

        composite_score = max(0.0, min(100.0, base_score))
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            quality_score=composite_score,
            errors=errors,
            warnings=warnings,
            grounding=grounding,
            parsed_json=parsed_json,
            was_repaired=was_repaired,
            details={
                "word_count": word_count,
                "expected_format": expected_format,
                "schema_used": schema_name or "none",
            }
        )
