"""
Prompt Engineering and Template Management Engine
Covers Summarization, Extraction, Classification, and Content Generation.
Supports few-shot demonstrations, role specialization, schema constraints, and chain-of-thought prompting.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class TaskCategory(str, Enum):
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    CONTENT_GENERATION = "content_generation"
    STRUCTURED_DATA = "structured_data"


@dataclass
class PromptTemplate:
    template_id: str
    name: str
    category: TaskCategory
    description: str
    system_role: str
    template_text: str
    variables: List[str]
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    output_format: str = "markdown"  # 'markdown', 'json', 'plain'
    enable_chain_of_thought: bool = False
    recommended_temperature: float = 0.2

    def render(self, values: Dict[str, Any], include_few_shots: bool = True) -> Dict[str, str]:
        """
        Renders the prompt text by replacing {{variable_name}} placeholders.
        Returns dictionary with 'system_prompt' and 'user_prompt'.
        """
        rendered_user = self.template_text
        for var_name in self.variables:
            val = str(values.get(var_name, f"[{var_name} not provided]"))
            rendered_user = rendered_user.replace(f"{{{{{var_name}}}}}", val)

        # Append few-shot examples if present and requested
        examples_block = ""
        if include_few_shots and self.few_shot_examples:
            examples_block = "\n\n### Reference Demonstrations (Few-Shot Examples):\n"
            for i, ex in enumerate(self.few_shot_examples, 1):
                examples_block += f"\n--- Example {i} ---\nInput: {ex.get('input', '')}\nReasoning: {ex.get('reasoning', 'Direct derivation.')}\nExpected Output:\n{ex.get('output', '')}\n"

        system_prompt = self.system_role
        if self.enable_chain_of_thought:
            system_prompt += "\n\nReasoning Instruction: Before providing the final answer, clearly show your step-by-step analytical reasoning inside a <reasoning>...</reasoning> block."

        if self.output_format == "json":
            system_prompt += "\n\nCRITICAL FORMAT REQUIREMENT: Respond ONLY with valid, parseable JSON conforming strictly to the requested schema. Do not include markdown code ticks (```json ... ```) or conversational commentary."

        final_user_prompt = rendered_user + examples_block

        return {
            "template_id": self.template_id,
            "category": self.category.value,
            "system_prompt": system_prompt.strip(),
            "user_prompt": final_user_prompt.strip(),
            "output_format": self.output_format,
            "temperature": self.recommended_temperature,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "system_role": self.system_role,
            "template_text": self.template_text,
            "variables": self.variables,
            "few_shot_examples": self.few_shot_examples,
            "output_format": self.output_format,
            "enable_chain_of_thought": self.enable_chain_of_thought,
            "recommended_temperature": self.recommended_temperature,
        }


class PromptEngine:
    """Manages built-in and user-customized prompt templates across document intelligence tasks."""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        # 1. SUMMARIZATION: Executive Summary
        self.register_template(PromptTemplate(
            template_id="sum_exec_brief",
            name="Executive Briefing & Strategic Summary",
            category=TaskCategory.SUMMARIZATION,
            description="Generates an executive-level synthesized briefing focusing on strategic objectives, key decisions, metrics, and risks.",
            system_role="You are a senior executive intelligence analyst. Synthesize complex documents into clear, high-impact executive summaries. Ground all claims strictly on the provided text.",
            template_text="""Please review the following document and synthesize an Executive Briefing based on user requirements.

USER REQUIREMENTS:
{{requirements}}

DOCUMENT CONTENT:
\"\"\"
{{document_text}}
\"\"\"

Please structure the briefing as follows:
# Executive Briefing: [Document Title / Topic]

## 1. Core Summary & Strategic Context
A concise paragraph capturing the primary premise and overarching objective.

## 2. Key Findings & Quantitative Metrics
- Bullet points highlighting verified facts, financial numbers, percentages, or milestones with direct context.

## 3. Critical Implications & Strategic Impact
- Analysis of what these findings mean for stakeholders, operations, or policy.

## 4. Recommended Action Items & Next Steps
- Concrete, prioritized next steps with suggested owners or timelines if mentioned in the text.
""",
            variables=["requirements", "document_text"],
            recommended_temperature=0.2,
            few_shot_examples=[
                {
                    "input": "Quarterly earnings preview indicating 14% YoY revenue growth to $4.2B with cloud margins expanding to 68%.",
                    "reasoning": "Extract core metric, calculate significance, isolate cloud margin trend.",
                    "output": "# Executive Briefing: Q3 Financial Trajectory\n\n## 1. Core Summary\nRobust top-line performance powered by cloud infrastructure margin expansion.\n\n## 2. Key Findings\n- Revenue reached $4.2B (+14% YoY).\n- Cloud gross margins expanded to 68%."
                }
            ]
        ))

        # 2. SUMMARIZATION: Multi-Perspective Analytical Digest
        self.register_template(PromptTemplate(
            template_id="sum_multi_perspective",
            name="Multi-Perspective Stakeholder Digest",
            category=TaskCategory.SUMMARIZATION,
            description="Examines the document through Technical, Financial, Operational, and Compliance viewpoints.",
            system_role="You are a multi-disciplinary document intelligence specialist capable of analyzing texts from engineering, financial, operational, and regulatory angles.",
            template_text="""Analyze the following document and produce a Multi-Perspective Digest addressing:
Requirements: {{requirements}}

DOCUMENT:
\"\"\"
{{document_text}}
\"\"\"

Format the output strictly with these distinct perspectives:
- **Technical & Architecture View**: Technical specifications, system architecture, dependencies.
- **Financial & Commercial View**: Budget, revenue impact, cost risks, contractual commitments.
- **Operations & Implementation View**: Deployment timeline, resource constraints, milestones.
- **Governance & Risk View**: Compliance, legal stipulations, audit concerns.
""",
            variables=["requirements", "document_text"],
            recommended_temperature=0.3
        ))

        # 3. EXTRACTION: Key Entities, Metrics & Timeline
        self.register_template(PromptTemplate(
            template_id="ext_entities_metrics",
            name="Structured Entity & KPI Extraction",
            category=TaskCategory.EXTRACTION,
            description="Extracts named entities (people, organizations, locations), financial/quantitative KPIs, dates, and contractual obligations.",
            system_role="You are a high-precision document extraction engine. Extract only facts that are explicitly stated in the document. Never speculate or hallucinate outside the given text.",
            template_text="""Extract all critical factual entities and metrics from the document according to user focus:
Focus: {{requirements}}

DOCUMENT CONTENT:
\"\"\"
{{document_text}}
\"\"\"

Present the extracted information in clean Markdown tables:

### 1. Key Organizations & Stakeholders
| Name | Entity Type | Role / Relationship | Context Mentioned |
|---|---|---|---|

### 2. Quantitative Metrics & KPIs
| Metric Name | Value | Unit / Currency | Period / Benchmark |
|---|---|---|---|

### 3. Dates, Deadlines & Milestones
| Date / Timeframe | Event / Deliverable | Criticality |
|---|---|---|

### 4. Obligations, Constraints & Risks
- List each identified obligation or constraint with exact paragraph/clause reference.
""",
            variables=["requirements", "document_text"],
            recommended_temperature=0.1
        ))

        # 4. EXTRACTION: Structured JSON Data Schema
        self.register_template(PromptTemplate(
            template_id="ext_json_schema",
            name="Strict JSON Schema Entity Extraction",
            category=TaskCategory.EXTRACTION,
            description="Extracts document facts directly into a validated JSON schema format.",
            system_role="You are an automated information extraction system that outputs strictly valid JSON without any markdown formatting or commentary.",
            template_text="""Extract data from the document into a strict JSON object following this schema:
{
  "document_summary": "string",
  "document_category": "string",
  "confidence_score": 0.95,
  "key_entities": [
    {"name": "string", "type": "ORGANIZATION|PERSON|LOCATION|TECHNOLOGY", "context": "string"}
  ],
  "metrics": [
    {"kpi": "string", "value": "string", "unit": "string"}
  ],
  "action_items": [
    {"task": "string", "priority": "HIGH|MEDIUM|LOW", "assigned_to": "string"}
  ]
}

SPECIFIC FOCUS: {{requirements}}

DOCUMENT:
\"\"\"
{{document_text}}
\"\"\"
""",
            variables=["requirements", "document_text"],
            output_format="json",
            recommended_temperature=0.1
        ))

        # 5. CLASSIFICATION: Taxonomy & Risk Assessment
        self.register_template(PromptTemplate(
            template_id="cls_taxonomy_risk",
            name="Document Classification & Risk Assessment",
            category=TaskCategory.CLASSIFICATION,
            description="Classifies document category, urgency, confidentiality level, and regulatory compliance risks.",
            system_role="You are an enterprise document governance classifier. You evaluate documents for categorization, confidentiality, urgency, and compliance exposure.",
            template_text="""Classify the document below based on requirements:
Requirements: {{requirements}}

DOCUMENT:
\"\"\"
{{document_text}}
\"\"\"

Provide the classification in this structure:

### Classification Verdict
- **Primary Domain**: [e.g. Legal Contract / Financial Report / Technical Architecture / Clinical Trial / HR Policy / General]
- **Document Sub-Type**: [Specific document type]
- **Urgency Level**: [LOW | MEDIUM | HIGH | CRITICAL] (with 1-sentence justification)
- **Confidentiality Level**: [PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED]
- **Sentiment & Tone**: [Objective / Cautionary / Optimistic / Critical / Formal]

### Compliance & Risk Matrix
| Risk Category | Risk Level (Low/Med/High) | Evidence From Text | Mitigation Note |
|---|---|---|---|
| Regulatory / Compliance | ... | ... | ... |
| Financial Exposure | ... | ... | ... |
| Operational Disruption | ... | ... | ... |
| Data Privacy / Security | ... | ... | ... |
""",
            variables=["requirements", "document_text"],
            recommended_temperature=0.2
        ))

        # 6. CONTENT GENERATION: Executive Action Memo & Roadmap
        self.register_template(PromptTemplate(
            template_id="gen_action_plan",
            name="Executive Action Memo & Implementation Roadmap",
            category=TaskCategory.CONTENT_GENERATION,
            description="Transforms document findings and user requirements into a comprehensive action memo, mitigation strategy, and phased implementation roadmap.",
            system_role="You are a management consultant and strategy director. Transform raw documents and strategic directives into clear, actionable, professional implementation plans.",
            template_text="""Generate a strategic Executive Action Memo based on the document and user requirements.

USER INSTRUCTIONS:
{{requirements}}

TARGET AUDIENCE:
{{target_audience}}

DOCUMENT CONTENT:
\"\"\"
{{document_text}}
\"\"\"

Create a comprehensive memo formatted with:
# Action Memo: Strategic Execution Roadmap

## Purpose & Background
Clear synthesis of the challenge or opportunity identified in the document.

## Key Insights Driving Strategy
The critical facts, numbers, and drivers that necessitate action.

## Phased Implementation Roadmap
- **Phase 1: Immediate Triage (Days 1 - 30)**: Milestones, key deliverables, owners.
- **Phase 2: Execution & Scaling (Days 31 - 90)**: Structural rollouts, process refinements.
- **Phase 3: Optimization & Long-term Governance (Days 91+)**: Continuous monitoring and KPI measurement.

## Risk Management & Guardrails
Anticipated barriers and proactive mitigation protocols.
""",
            variables=["requirements", "target_audience", "document_text"],
            recommended_temperature=0.3
        ))

        # 7. CONTENT GENERATION: Intelligent Q&A / Knowledge FAQ
        self.register_template(PromptTemplate(
            template_id="gen_faq_knowledge",
            name="Knowledge Base & High-Impact FAQ Generation",
            category=TaskCategory.CONTENT_GENERATION,
            description="Generates an exhaustive, high-yield FAQ and troubleshooting knowledge base derived directly from the document.",
            system_role="You are a knowledge architect and customer education specialist. Formulate comprehensive, accurate FAQs directly grounded in the provided reference text.",
            template_text="""Generate a structured Knowledge Base and FAQ based on the document.
Focus areas: {{requirements}}

DOCUMENT:
\"\"\"
{{document_text}}
\"\"\"

Generate 5-8 in-depth question and answer pairs covering:
1. Core definitions and purpose
2. Nuanced rules, caveats, or conditional policies
3. Numerical thresholds or quantitative metrics
4. Exception handling and troubleshooting steps

Format each entry as:
### Q: [Specific, natural user question]
**Answer**: [Accurate, grounded explanation with direct evidence from the document.]
*Source Reference*: [Mention which section or metric supports this answer.]
""",
            variables=["requirements", "document_text"],
            recommended_temperature=0.3
        ))

        # 8. STRUCTURED DATA: Pydantic Validation Template
        self.register_template(PromptTemplate(
            template_id="struct_pydantic_eval",
            name="Standard Document Intelligence Schema",
            category=TaskCategory.STRUCTURED_DATA,
            description="Extracts data matching the platform's core Pydantic DocumentIntelligenceModel.",
            system_role="You are a machine-readable data serialization engine. You strictly output JSON matching the required schema.",
            template_text="""Extract all relevant intelligence into this exact JSON structure:
{
  "title": "string",
  "document_type": "string",
  "summary": "string",
  "key_findings": ["string"],
  "entities": [
    {"name": "string", "category": "string", "relevance": "string"}
  ],
  "metrics": [
    {"label": "string", "value": "string", "unit": "string"}
  ],
  "risk_assessment": {
    "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
    "risk_factors": ["string"]
  },
  "action_items": [
    {"task": "string", "priority": "HIGH|MEDIUM|LOW"}
  ]
}

USER CRITERIA: {{requirements}}

SOURCE TEXT:
\"\"\"
{{document_text}}
\"\"\"
""",
            variables=["requirements", "document_text"],
            output_format="json",
            recommended_temperature=0.1
        ))

    def register_template(self, template: PromptTemplate):
        """Registers a new or updated prompt template."""
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Retrieves a template by id."""
        return self._templates.get(template_id)

    def list_templates(self, category: Optional[TaskCategory] = None) -> List[Dict[str, Any]]:
        """Lists all registered templates, optionally filtered by category."""
        templates = self._templates.values()
        if category:
            templates = [t for t in templates if t.category == category]
        return [t.to_dict() for t in templates]

    def render(self, template_id: str, values: Dict[str, Any], include_few_shots: bool = True) -> Dict[str, str]:
        """Renders the specified template with the given parameter values."""
        tmpl = self.get_template(template_id)
        if not tmpl:
            raise KeyError(f"Template '{template_id}' not found. Available: {list(self._templates.keys())}")
        return tmpl.render(values, include_few_shots=include_few_shots)
