"""
DocIntel & GenAI Platform - Command Line Interface (CLI)
Enables command-line batch document ingestion, prompt template execution,
multi-step pipeline runs, validation checks, and evaluation benchmarking.
"""

import sys
import argparse
import json
from pathlib import Path

from core.ingestion import DocumentIngestor
from core.prompt_engine import PromptEngine
from core.llm_client import LLMClient
from core.validator import OutputValidator
from core.evaluation import EvaluationEngine
from core.workflow_orchestrator import WorkflowOrchestrator


def print_banner():
    print("=" * 72)
    print("  AI-POWERED GENERATIVE CONTENT & DOCUMENT INTELLIGENCE PLATFORM")
    print("=" * 72)


def cmd_ingest(args):
    ingestor = DocumentIngestor()
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    print(f"\n[*] Ingesting file: {path.name} (strategy={args.strategy}, chunk_size={args.chunk_size})...")
    doc = ingestor.ingest_file(
        str(path),
        chunk_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    print(f"\n[+] Ingestion Successful:")
    print(f"    - Document ID:    {doc.doc_id}")
    print(f"    - File Type:      {doc.file_type}")
    print(f"    - Clean Length:   {len(doc.clean_text)} characters")
    print(f"    - Total Words:    {doc.statistics['word_count']}")
    print(f"    - Est. Tokens:    {doc.statistics['estimated_tokens']}")
    print(f"    - Chunks Created: {len(doc.chunks)}")
    print(f"    - Sections Found: {len(doc.sections)}")

    print("\n--- Top Keywords ---")
    for kw in doc.statistics["top_keywords"][:6]:
        print(f"  • {kw['word']}: {kw['frequency']}")

    if args.json:
        print("\n" + json.dumps(doc.to_dict(), indent=2))


def cmd_list_templates(args):
    engine = PromptEngine()
    templates = engine.list_templates()
    print(f"\nAvailable Prompt Templates ({len(templates)} registered):\n")
    for t in templates:
        print(f"  • [{t['template_id']}] {t['name']}")
        print(f"    Category:    {t['category']}")
        print(f"    Description: {t['description']}")
        print(f"    Variables:   {', '.join(t['variables'])}\n")


def cmd_list_pipelines(args):
    orchestrator = WorkflowOrchestrator()
    pipelines = orchestrator.list_pipelines()
    print(f"\nAvailable Workflow Pipelines ({len(pipelines)} registered):\n")
    for p in pipelines:
        print(f"  • [{p['pipeline_id']}] {p['name']}")
        print(f"    Description: {p['description']}")
        print(f"    Steps ({p['steps_count']}):")
        for s in p["steps"]:
            print(f"      - {s['step_id']}: {s['name']} ({s['action_type']})")
        print()


def cmd_run_prompt(args):
    ingestor = DocumentIngestor()
    prompt_engine = PromptEngine()
    llm = LLMClient(mode=args.mode)

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    doc = ingestor.ingest_file(str(path))
    template = prompt_engine.get_template(args.template)
    if not template:
        print(f"Error: Template '{args.template}' not found. Use 'list-templates' to inspect.")
        sys.exit(1)

    rendered = prompt_engine.render(args.template, {
        "requirements": args.requirements,
        "target_audience": args.audience,
        "document_text": doc.clean_text[:5000]
    })

    print(f"\n[*] Executing Prompt '{template.name}' using mode '{args.mode}'...")
    response = llm.generate(
        prompt=rendered["user_prompt"],
        system_prompt=rendered["system_prompt"],
        temperature=rendered["temperature"],
        output_format=rendered["output_format"],
        category=rendered["category"]
    )

    print(f"\n[+] Generated in {response.latency_ms:.1f}ms (Provider: {response.provider}, Model: {response.model_name}):\n")
    print(response.content)

    if args.validate:
        validator = OutputValidator()
        val_res = validator.validate_output(response.content, source_text=doc.clean_text)
        print("\n" + "=" * 50)
        print(f"VALIDATION REPORT: Valid={val_res.is_valid} | Score={val_res.quality_score}% | Grounding={round(val_res.grounding.grounding_ratio*100, 1)}%")
        if val_res.warnings:
            for w in val_res.warnings:
                print(f"  Warning: {w.message}")

    if args.evaluate:
        evaluator = EvaluationEngine(llm_client=llm)
        eval_rep = evaluator.evaluate(response.content, source_text=doc.clean_text, requirements=args.requirements)
        print("\n" + "=" * 50)
        print(f"EVALUATION BENCHMARK: Index={eval_rep.overall_quality_index:.1f}% ({eval_rep.grade})")
        print(f"  • Factuality:   {eval_rep.factuality.percentage:.1f}% ({eval_rep.factuality.rating})")
        print(f"  • Relevance:    {eval_rep.relevance.percentage:.1f}% ({eval_rep.relevance.rating})")
        print(f"  • Consistency:  {eval_rep.consistency.percentage:.1f}% ({eval_rep.consistency.rating})")
        print(f"  • Completeness: {eval_rep.completeness.percentage:.1f}% ({eval_rep.completeness.rating})")
        print(f"  • Readability:  {eval_rep.readability.percentage:.1f}% ({eval_rep.readability.rating})")


def cmd_run_pipeline(args):
    ingestor = DocumentIngestor()
    orchestrator = WorkflowOrchestrator()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    print(f"\n[*] Ingesting: {path.name}...")
    doc = ingestor.ingest_file(str(path))

    print(f"[*] Executing Pipeline: {args.pipeline}...")

    def progress_callback(data):
        st = data.get("step", {})
        status = data.get("status", "")
        if status == "running":
            print(f"  -> Running step: [{st.get('step_id')}] {st.get('name')}...")
        elif status == "completed":
            print(f"  ✓ Finished step: [{st.get('step_id')}] in {st.get('duration_ms')}ms")

    context = orchestrator.execute_pipeline(
        pipeline_id=args.pipeline,
        document=doc,
        user_requirements=args.requirements,
        target_audience=args.audience,
        progress_callback=progress_callback
    )

    print("\n" + "=" * 72)
    print(f"WORKFLOW COMPLETED: {context.workflow_id}")
    print("=" * 72)

    if context.final_output:
        print("\n### FINAL SYNTHESIS:\n")
        print(context.final_output)

    if context.validation_result:
        vr = context.validation_result
        print("\n### VALIDATION STATUS:")
        print(f"  Valid: {vr.is_valid} | Quality Score: {vr.quality_score}% | Grounding: {round(vr.grounding.grounding_ratio*100, 1)}%")

    if context.evaluation_report:
        er = context.evaluation_report
        print("\n### MULTI-DIMENSIONAL BENCHMARK:")
        print(f"  Quality Index: {er.overall_quality_index:.1f}% (Grade: {er.grade})")
        for k, v in er.radar_data.items():
            print(f"    - {k:12}: {v:.1f}%")

    if args.output_file:
        out_p = Path(args.output_file)
        out_p.write_text(context.final_output, encoding="utf-8")
        print(f"\n[+] Saved final output to: {out_p.resolve()}")


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="AI-Powered Generative Content & Document Intelligence Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest and preprocess a document file")
    p_ingest.add_argument("file", help="Path to document file (pdf, docx, txt, csv, md)")
    p_ingest.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed_window", "section"])
    p_ingest.add_argument("--chunk-size", type=int, default=500)
    p_ingest.add_argument("--chunk-overlap", type=int, default=100)
    p_ingest.add_argument("--json", action="store_true", help="Print full JSON output")
    p_ingest.set_defaults(func=cmd_ingest)

    # List templates
    p_templates = subparsers.add_parser("list-templates", help="List registered prompt templates")
    p_templates.set_defaults(func=cmd_list_templates)

    # List pipelines
    p_pipelines = subparsers.add_parser("list-pipelines", help="List available workflow pipelines")
    p_pipelines.set_defaults(func=cmd_list_pipelines)

    # Run prompt
    p_prompt = subparsers.add_parser("run-prompt", help="Execute a prompt template on a document")
    p_prompt.add_argument("file", help="Path to document file")
    p_prompt.add_argument("--template", default="sum_exec_brief", help="Template ID")
    p_prompt.add_argument("--requirements", default="Synthesize key findings and strategic impacts.")
    p_prompt.add_argument("--audience", default="Executive Stakeholders")
    p_prompt.add_argument("--mode", default="auto", choices=["auto", "gemini", "local"])
    p_prompt.add_argument("--validate", action="store_true", help="Run output validation")
    p_prompt.add_argument("--evaluate", action="store_true", help="Run evaluation benchmark")
    p_prompt.set_defaults(func=cmd_run_prompt)

    # Run pipeline
    p_pipe = subparsers.add_parser("run-pipeline", help="Execute multi-step workflow pipeline")
    p_pipe.add_argument("file", help="Path to document file")
    p_pipe.add_argument("--pipeline", default="full_doc_intel", help="Pipeline ID")
    p_pipe.add_argument("--requirements", default="Synthesize key findings, metrics, and risk profile.")
    p_pipe.add_argument("--audience", default="Executive Leadership")
    p_pipe.add_argument("--output-file", help="Path to save final output")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
