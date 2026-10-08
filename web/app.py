"""
DocIntel & GenAI Platform - Web Application & REST API
Flask server providing endpoints for Ingestion, Prompt Engineering,
Multi-Step Workflows, Schema Validation, and Output Evaluation.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure core package is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ingestion import DocumentIngestor, IngestedDocument
from core.prompt_engine import PromptEngine, PromptTemplate, TaskCategory
from core.llm_client import LLMClient
from core.validator import OutputValidator
from core.evaluation import EvaluationEngine
from core.workflow_orchestrator import WorkflowOrchestrator

load_dotenv(BASE_DIR / ".env")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static")
)
CORS(app)

# Initialize Core Services
ingestor = DocumentIngestor()
prompt_engine = PromptEngine()
llm_client = LLMClient(mode=os.environ.get("ENGINE_MODE", "auto"))
validator = OutputValidator()
eval_engine = EvaluationEngine(llm_client=llm_client)
orchestrator = WorkflowOrchestrator(
    ingestor=ingestor,
    prompt_engine=prompt_engine,
    llm_client=llm_client,
    validator=validator,
    evaluation_engine=eval_engine
)

# In-memory session store for current document and workflow runs
SESSION_STATE = {
    "current_document": None,
    "last_workflow_context": None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    has_key = bool(llm_client.api_key)
    return jsonify({
        "status": "online",
        "has_api_key": has_key,
        "active_model": llm_client.preferred_model,
        "engine_mode": llm_client.mode,
        "available_samples": len(list((BASE_DIR / "data" / "samples").glob("*.txt"))),
        "available_templates": len(prompt_engine.list_templates()),
        "available_pipelines": len(orchestrator.list_pipelines()),
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json() or {}
    api_key = data.get("api_key")
    mode = data.get("mode")
    model = data.get("model")

    if api_key is not None:
        llm_client.set_api_key(api_key)
    if mode in ["auto", "gemini", "local"]:
        llm_client.mode = mode
    if model:
        llm_client.preferred_model = model

    return jsonify({
        "success": True,
        "has_api_key": bool(llm_client.api_key),
        "engine_mode": llm_client.mode,
        "active_model": llm_client.preferred_model,
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    samples_dir = BASE_DIR / "data" / "samples"
    samples = []
    for f in samples_dir.glob("*.txt"):
        samples.append({
            "filename": f.name,
            "title": f.stem.replace("_", " ").title(),
            "size_kb": round(f.stat().st_size / 1024, 2)
        })
    return jsonify({"samples": samples})


@app.route("/api/samples/<filename>", methods=["GET"])
def load_sample(filename):
    samples_dir = BASE_DIR / "data" / "samples"
    file_path = samples_dir / filename
    if not file_path.exists():
        return jsonify({"error": f"Sample '{filename}' not found"}), 404

    strategy = request.args.get("strategy", "paragraph")
    chunk_size = int(request.args.get("chunk_size", 500))
    chunk_overlap = int(request.args.get("chunk_overlap", 100))

    doc = ingestor.ingest_file(
        str(file_path),
        chunk_strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    SESSION_STATE["current_document"] = doc

    return jsonify({
        "success": True,
        "document": doc.to_dict(),
        "clean_text": doc.clean_text
    })


@app.route("/api/ingest", methods=["POST"])
def ingest_document():
    strategy = request.form.get("chunk_strategy", "paragraph")
    chunk_size = int(request.form.get("chunk_size", 500))
    chunk_overlap = int(request.form.get("chunk_overlap", 100))

    if "file" in request.files:
        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            return jsonify({"error": "No selected file"}), 400

        file_bytes = uploaded_file.read()
        doc = ingestor.ingest_bytes(
            file_bytes=file_bytes,
            filename=uploaded_file.filename,
            chunk_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif request.is_json:
        data = request.get_json()
        raw_text = data.get("text", "")
        filename = data.get("filename", "manual_input.txt")
        if not raw_text.strip():
            return jsonify({"error": "Text content cannot be empty"}), 400

        doc = ingestor.ingest_direct_text(
            text=raw_text,
            filename=filename,
            chunk_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        return jsonify({"error": "No file or text provided"}), 400

    SESSION_STATE["current_document"] = doc

    return jsonify({
        "success": True,
        "document": doc.to_dict(),
        "clean_text": doc.clean_text
    })


@app.route("/api/templates", methods=["GET"])
def get_templates():
    category = request.args.get("category")
    cat_enum = TaskCategory(category) if category else None
    templates = prompt_engine.list_templates(category=cat_enum)
    return jsonify({"templates": templates})


@app.route("/api/prompt/generate", methods=["POST"])
def generate_prompt_output():
    data = request.get_json() or {}
    template_id = data.get("template_id")
    requirements = data.get("requirements", "")
    target_audience = data.get("target_audience", "General")
    custom_system_prompt = data.get("custom_system_prompt")
    custom_user_prompt = data.get("custom_user_prompt")
    temperature = float(data.get("temperature", 0.2))

    doc = SESSION_STATE.get("current_document")
    doc_text = data.get("document_text") or (doc.clean_text if doc else "")

    if not doc_text and not custom_user_prompt:
        return jsonify({"error": "No document loaded. Please ingest a document first."}), 400

    # Render template if template_id provided
    if template_id:
        template = prompt_engine.get_template(template_id)
        if not template:
            return jsonify({"error": f"Template '{template_id}' not found"}), 404

        rendered = prompt_engine.render(template_id, {
            "requirements": requirements,
            "target_audience": target_audience,
            "document_text": doc_text
        })
        sys_prompt = custom_system_prompt or rendered["system_prompt"]
        usr_prompt = rendered["user_prompt"]
        out_fmt = rendered["output_format"]
        category = rendered["category"]
    else:
        sys_prompt = custom_system_prompt or "You are an expert document intelligence assistant."
        usr_prompt = custom_user_prompt or f"Analyze the following text based on: {requirements}\n\n{doc_text}"
        out_fmt = data.get("output_format", "markdown")
        category = "general"

    response = llm_client.generate(
        prompt=usr_prompt,
        system_prompt=sys_prompt,
        temperature=temperature,
        output_format=out_fmt,
        category=category
    )

    return jsonify({
        "success": True,
        "response": response.to_dict(),
        "rendered_prompt": {
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            "output_format": out_fmt,
        }
    })


@app.route("/api/validate", methods=["POST"])
def validate_output():
    data = request.get_json() or {}
    output_text = data.get("output_text", "")
    source_text = data.get("source_text")
    expected_format = data.get("expected_format", "markdown")
    schema_name = data.get("schema_name")

    if not source_text and SESSION_STATE.get("current_document"):
        source_text = SESSION_STATE["current_document"].clean_text

    val_res = validator.validate_output(
        generated_output=output_text,
        source_text=source_text or "",
        expected_format=expected_format,
        schema_name=schema_name
    )

    return jsonify({
        "success": True,
        "validation": val_res.to_dict()
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate_output():
    data = request.get_json() or {}
    output_text = data.get("output_text", "")
    source_text = data.get("source_text")
    requirements = data.get("requirements", "")
    run_llm_judge = bool(data.get("run_llm_judge", False))

    if not source_text and SESSION_STATE.get("current_document"):
        source_text = SESSION_STATE["current_document"].clean_text

    eval_rep = eval_engine.evaluate(
        generated_text=output_text,
        source_text=source_text or "",
        requirements=requirements,
        run_llm_judge=run_llm_judge
    )

    return jsonify({
        "success": True,
        "evaluation": eval_rep.to_dict()
    })


@app.route("/api/pipelines", methods=["GET"])
def get_pipelines():
    pipelines = orchestrator.list_pipelines()
    return jsonify({"pipelines": pipelines})


@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    data = request.get_json() or {}
    pipeline_id = data.get("pipeline_id", "full_doc_intel")
    requirements = data.get("requirements", "Synthesize findings and identify core action points.")
    target_audience = data.get("target_audience", "Executive Stakeholders")

    doc = SESSION_STATE.get("current_document")
    if not doc:
        return jsonify({"error": "No document is currently loaded. Please upload or load a sample first."}), 400

    context = orchestrator.execute_pipeline(
        pipeline_id=pipeline_id,
        document=doc,
        user_requirements=requirements,
        target_audience=target_audience
    )
    SESSION_STATE["last_workflow_context"] = context

    return jsonify({
        "success": True,
        "workflow": context.to_dict(),
        "intermediate_artifacts": context.intermediate_artifacts,
        "step_history": [s.to_dict() for s in context.step_history],
        "final_output": context.final_output,
        "validation_result": context.validation_result.to_dict() if context.validation_result else None,
        "evaluation_report": context.evaluation_report.to_dict() if context.evaluation_report else None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"[*] Starting DocIntel & GenAI Platform on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
