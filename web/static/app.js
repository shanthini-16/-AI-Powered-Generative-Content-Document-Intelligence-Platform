// DocIntel & GenAI Platform Frontend Application Logic

let appState = {
    currentDocument: null,
    templates: [],
    pipelines: [],
    currentOutput: "",
    lastValidation: null,
    lastEvaluation: null,
    radarChart: null,
    activePipelineContext: null,
};

document.addEventListener("DOMContentLoaded", () => {
    initApp();
    setupDropZone();
});

// =====================================================================
// Initialization & Tab Navigation
// =====================================================================

async function initApp() {
    await fetchStatus();
    await loadSamples();
    await loadTemplates();
    await loadPipelines();
    initRadarChart();
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });

    document.querySelectorAll(".tab-content").forEach(sec => {
        sec.classList.remove("active");
    });

    const targetSec = document.getElementById(`tab-${tabId}`);
    if (targetSec) {
        targetSec.classList.add("active");
    }

    // Refresh icons on tab change
    if (window.lucide) {
        lucide.createIcons();
    }

    // If switching to evaluation, resize chart
    if (tabId === "evaluation" && appState.radarChart) {
        setTimeout(() => appState.radarChart.resize(), 100);
    }
}

async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        const badge = document.getElementById("statusEngineBadge");
        const statusText = document.getElementById("engineStatusText");

        if (data.has_api_key) {
            badge.className = "flex items-center space-x-2 text-xs font-medium px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
            statusText.innerText = `Gemini (${data.active_model})`;
        } else {
            badge.className = "flex items-center space-x-2 text-xs font-medium px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20";
            statusText.innerText = "Offline NLP Engine";
        }
    } catch (err) {
        console.error("Failed to fetch system status:", err);
    }
}

// =====================================================================
// Sample Documents & Ingestion
// =====================================================================

async function loadSamples() {
    try {
        const res = await fetch("/api/samples");
        const data = await res.json();
        const grid = document.getElementById("samplesGrid");
        grid.innerHTML = "";

        data.samples.forEach(s => {
            const card = document.createElement("div");
            card.className = "card p-4 hover:border-teal-500/50 cursor-pointer flex flex-col justify-between";
            card.onclick = () => loadSampleDoc(s.filename);
            card.innerHTML = `
                <div>
                    <div class="flex items-center space-x-2 text-teal-400 mb-2">
                        <i data-lucide="file-text" class="w-4 h-4"></i>
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400">${s.size_kb} KB</span>
                    </div>
                    <h4 class="text-sm font-semibold text-white">${s.title}</h4>
                </div>
                <button class="mt-3 text-xs text-teal-400 hover:text-teal-300 font-semibold flex items-center space-x-1">
                    <span>Load & Ingest</span>
                    <i data-lucide="arrow-right" class="w-3 h-3"></i>
                </button>
            `;
            grid.appendChild(card);
        });
        if (window.lucide) lucide.createIcons();
    } catch (err) {
        console.error("Error loading samples:", err);
    }
}

async function loadSampleDoc(filename) {
    try {
        showLoading(true, `Ingesting and profiling '${filename}'...`);
        const strat = document.getElementById("chunkStrategySelect")?.value || "paragraph";
        const cSize = document.getElementById("chunkSizeInput")?.value || 500;
        const cOver = document.getElementById("chunkOverlapInput")?.value || 100;

        const res = await fetch(`/api/samples/${filename}?strategy=${strat}&chunk_size=${cSize}&chunk_overlap=${cOver}`);
        const data = await res.json();
        if (data.success) {
            handleDocumentLoaded(data.document, data.clean_text);
            switchTab("ingest");
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (err) {
        alert(`Failed to load sample: ${err}`);
    } finally {
        showLoading(false);
    }
}

function setupDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone.onclick = () => fileInput.click();

    fileInput.onchange = (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    };

    dropZone.ondragover = (e) => {
        e.preventDefault();
        dropZone.classList.add("border-indigo-500", "bg-indigo-950/30");
    };

    dropZone.ondragleave = () => {
        dropZone.classList.remove("border-indigo-500", "bg-indigo-950/30");
    };

    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-indigo-500", "bg-indigo-950/30");
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    };
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("chunk_strategy", document.getElementById("chunkStrategySelect").value);
    formData.append("chunk_size", document.getElementById("chunkSizeInput").value);
    formData.append("chunk_overlap", document.getElementById("chunkOverlapInput").value);

    try {
        showLoading(true, `Ingesting and parsing ${file.name}...`);
        const res = await fetch("/api/ingest", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        if (data.success) {
            handleDocumentLoaded(data.document, data.clean_text);
        } else {
            alert(`Ingestion failed: ${data.error}`);
        }
    } catch (err) {
        alert(`Upload error: ${err}`);
    } finally {
        showLoading(false);
    }
}

async function ingestDirectText() {
    const text = document.getElementById("rawTextInput").value;
    if (!text.trim()) {
        alert("Please paste document text first.");
        return;
    }

    try {
        showLoading(true, "Ingesting raw text...");
        const res = await fetch("/api/ingest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                filename: "pasted_document.txt",
                chunk_strategy: document.getElementById("chunkStrategySelect").value,
                chunk_size: parseInt(document.getElementById("chunkSizeInput").value),
                chunk_overlap: parseInt(document.getElementById("chunkOverlapInput").value),
            }),
        });
        const data = await res.json();
        if (data.success) {
            handleDocumentLoaded(data.document, data.clean_text);
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (err) {
        alert(`Ingestion error: ${err}`);
    } finally {
        showLoading(false);
    }
}

function handleDocumentLoaded(doc, cleanText) {
    appState.currentDocument = { ...doc, clean_text: cleanText };

    // Update active badges
    const badge = document.getElementById("loadedDocBadge");
    badge.classList.remove("hidden");
    document.getElementById("loadedDocName").innerText = doc.filename;
    document.getElementById("docTypeBadge").innerText = `${doc.file_type} • ID: ${doc.doc_id}`;

    // Update Stats
    document.getElementById("statWords").innerText = doc.statistics.word_count.toLocaleString();
    document.getElementById("statTokens").innerText = doc.statistics.estimated_tokens.toLocaleString();
    document.getElementById("statChunks").innerText = doc.chunks_count;
    document.getElementById("statDiversity").innerText = doc.statistics.lexical_diversity;
    document.getElementById("chunkCountPill").innerText = doc.chunks_count;

    // Populate Clean text
    document.getElementById("subViewClean").innerText = cleanText;

    // Populate Chunks list
    const chunksContainer = document.getElementById("subViewChunks");
    chunksContainer.innerHTML = "";
    doc.chunks.forEach(c => {
        const div = document.createElement("div");
        div.className = "p-3 bg-slate-900/80 rounded-lg border border-slate-800 text-xs";
        div.innerHTML = `
            <div class="flex justify-between items-center text-slate-400 mb-1">
                <span class="font-bold text-teal-400">Chunk #${c.chunk_id}</span>
                <span>${c.word_count} words (~${c.token_estimate} tokens)</span>
            </div>
            <p class="font-mono text-slate-300 whitespace-pre-wrap">${escapeHtml(c.text)}</p>
        `;
        chunksContainer.appendChild(div);
    });

    // Populate Keywords & Sections
    const kwContainer = document.getElementById("subViewKeywords");
    kwContainer.innerHTML = `
        <div>
            <h5 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Top Lexical Keywords</h5>
            <div class="flex flex-wrap gap-2">
                ${doc.statistics.top_keywords.map(k => `
                    <span class="px-2.5 py-1 rounded-md bg-teal-500/10 border border-teal-500/20 text-teal-300 text-xs font-mono">
                        ${k.word} <strong class="text-teal-400">(${k.frequency})</strong>
                    </span>
                `).join("")}
            </div>
        </div>
        <div class="mt-4 pt-4 border-t border-slate-800">
            <h5 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Identified Document Sections (${doc.sections.length})</h5>
            <div class="space-y-2">
                ${doc.sections.map(s => `
                    <div class="p-2.5 bg-slate-900/60 rounded border border-slate-800 text-xs">
                        <span class="font-bold text-white">${escapeHtml(s.title)}</span>
                        <span class="text-slate-400 ml-2">(${s.length} chars)</span>
                    </div>
                `).join("")}
            </div>
        </div>
    `;

    setDocViewMode("clean");
}

function setDocViewMode(mode) {
    document.getElementById("subViewClean").classList.toggle("hidden", mode !== "clean");
    document.getElementById("subViewChunks").classList.toggle("hidden", mode !== "chunks");
    document.getElementById("subViewKeywords").classList.toggle("hidden", mode !== "keywords");

    document.getElementById("viewModeCleanBtn").className = mode === "clean" ? "px-3 py-1 rounded-md text-xs font-semibold bg-teal-600 text-white shadow-sm shadow-teal-500/20" : "px-3 py-1 rounded-md text-xs font-semibold bg-slate-800/80 text-slate-300";
    document.getElementById("viewModeChunksBtn").className = mode === "chunks" ? "px-3 py-1 rounded-md text-xs font-semibold bg-teal-600 text-white shadow-sm shadow-teal-500/20" : "px-3 py-1 rounded-md text-xs font-semibold bg-slate-800/80 text-slate-300";
    document.getElementById("viewModeKeywordsBtn").className = mode === "keywords" ? "px-3 py-1 rounded-md text-xs font-semibold bg-teal-600 text-white shadow-sm shadow-teal-500/20" : "px-3 py-1 rounded-md text-xs font-semibold bg-slate-800/80 text-slate-300";
}


// =====================================================================
// Prompt Studio & Generation
// =====================================================================

async function loadTemplates() {
    try {
        const res = await fetch("/api/templates");
        const data = await res.json();
        appState.templates = data.templates;

        const select = document.getElementById("templateSelect");
        select.innerHTML = "";

        data.templates.forEach(t => {
            const opt = document.createElement("option");
            opt.value = t.template_id;
            opt.innerText = `[${t.category.toUpperCase()}] ${t.name}`;
            select.appendChild(opt);
        });

        onTemplateSelected();
    } catch (err) {
        console.error("Failed to load templates:", err);
    }
}

function onTemplateSelected() {
    const select = document.getElementById("templateSelect");
    const tmpl = appState.templates.find(t => t.template_id === select.value);
    if (!tmpl) return;

    document.getElementById("templateDesc").innerText = tmpl.description;
    document.getElementById("customSystemPrompt").value = tmpl.system_role;
    document.getElementById("tempSlider").value = tmpl.recommended_temperature;
    document.getElementById("tempVal").innerText = tmpl.recommended_temperature;

    if (tmpl.few_shot_examples && tmpl.few_shot_examples.length > 0) {
        document.getElementById("promptReqInput").placeholder = "Example: Focus on key metrics, compliance exposure, or strategic actions...";
    }
}

function toggleSystemPromptAccordion() {
    const container = document.getElementById("sysPromptContainer");
    const chevron = document.getElementById("sysPromptChevron");
    const isHidden = container.classList.contains("hidden");
    container.classList.toggle("hidden", !isHidden);
    chevron.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
}

async function executePromptStudio() {
    if (!appState.currentDocument) {
        alert("Please ingest or select a document first (via Dashboard or Document Ingestion tab).");
        switchTab("dashboard");
        return;
    }

    const templateId = document.getElementById("templateSelect").value;
    const req = document.getElementById("promptReqInput").value;
    const aud = document.getElementById("promptAudienceInput").value;
    const temp = parseFloat(document.getElementById("tempSlider").value);
    const mode = document.getElementById("promptModeSelect").value;
    const sysPrompt = document.getElementById("customSystemPrompt").value;

    const runBtn = document.getElementById("runPromptBtn");
    runBtn.disabled = true;
    runBtn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Generating Response...</span>`;
    if (window.lucide) lucide.createIcons();

    try {
        const res = await fetch("/api/prompt/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                template_id: templateId,
                requirements: req,
                target_audience: aud,
                custom_system_prompt: sysPrompt,
                temperature: temp,
                mode: mode,
            }),
        });
        const data = await res.json();
        if (data.success) {
            appState.currentOutput = data.response.content;
            document.getElementById("promptOutputArea").innerText = data.response.content;

            const metaBadge = document.getElementById("execMetaBadge");
            metaBadge.classList.remove("hidden");
            metaBadge.innerText = `${data.response.provider} • ${data.response.latency_ms}ms • ~${data.response.total_tokens} tokens`;
        } else {
            alert(`Generation Error: ${data.error}`);
        }
    } catch (err) {
        alert(`Execution error: ${err}`);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i><span>Generate Response</span>`;
        if (window.lucide) lucide.createIcons();
    }
}


// =====================================================================
// Context-Aware Multi-Step Pipelines
// =====================================================================

async function loadPipelines() {
    try {
        const res = await fetch("/api/pipelines");
        const data = await res.json();
        appState.pipelines = data.pipelines;

        const select = document.getElementById("pipelineSelect");
        select.innerHTML = "";

        data.pipelines.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.pipeline_id;
            opt.innerText = p.name;
            select.appendChild(opt);
        });

        onPipelineSelected();
    } catch (err) {
        console.error("Failed to load pipelines:", err);
    }
}

function onPipelineSelected() {
    const select = document.getElementById("pipelineSelect");
    const pipe = appState.pipelines.find(p => p.pipeline_id === select.value);
    if (!pipe) return;

    document.getElementById("pipelineDesc").innerText = pipe.description;

    const list = document.getElementById("pipelineStepsList");
    list.innerHTML = "";

    pipe.steps.forEach((s, idx) => {
        const stepDiv = document.createElement("div");
        stepDiv.id = `pipeStep_${s.step_id}`;
        stepDiv.className = "flex items-start space-x-3 p-3 bg-slate-900/60 rounded-lg border border-slate-800 text-xs";
        stepDiv.innerHTML = `
            <div class="h-6 w-6 rounded-full bg-slate-800 text-slate-400 font-bold flex items-center justify-center shrink-0 step-num">
                ${idx + 1}
            </div>
            <div class="flex-1">
                <span class="font-bold text-white block">${s.name}</span>
                <span class="text-slate-400 text-[11px]">${s.description}</span>
            </div>
            <div class="step-status text-slate-500">
                <i data-lucide="circle" class="w-4 h-4"></i>
            </div>
        `;
        list.appendChild(stepDiv);
    });

    if (window.lucide) lucide.createIcons();
}

async function executePipeline() {
    if (!appState.currentDocument) {
        alert("Please ingest a document first.");
        switchTab("dashboard");
        return;
    }

    const pipelineId = document.getElementById("pipelineSelect").value;
    const req = document.getElementById("pipelineReqInput").value;
    const runBtn = document.getElementById("runPipelineBtn");

    runBtn.disabled = true;
    runBtn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Running Pipeline...</span>`;
    if (window.lucide) lucide.createIcons();

    const progressCard = document.getElementById("pipelineProgressCard");
    progressCard.classList.remove("hidden");
    const progressBar = document.getElementById("pipelineProgressBar");
    const progressPct = document.getElementById("pipelineProgressPct");
    progressBar.style.width = "20%";
    progressPct.innerText = "20%";

    try {
        const res = await fetch("/api/pipeline/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pipeline_id: pipelineId,
                requirements: req,
            }),
        });
        const data = await res.json();
        if (data.success) {
            appState.activePipelineContext = data;
            appState.currentOutput = data.final_output;
            appState.lastValidation = data.validation_result;
            appState.lastEvaluation = data.evaluation_report;

            progressBar.style.width = "100%";
            progressPct.innerText = "100%";
            document.getElementById("pipelineProgressLabel").innerText = "Workflow Pipeline Completed Successfully!";

            // Render intermediate artifact tabs
            renderArtifactTabs(data);
            displayPipelineArtifact("final");

            // Update step status indicators
            data.step_history.forEach(st => {
                const stepElem = document.getElementById(`pipeStep_${st.step_id}`);
                if (stepElem) {
                    const statusIcon = stepElem.querySelector(".step-status");
                    statusIcon.innerHTML = `<i data-lucide="check-circle-2" class="w-4 h-4 text-emerald-400"></i>`;
                }
            });
            if (window.lucide) lucide.createIcons();
        } else {
            alert(`Pipeline error: ${data.error}`);
        }
    } catch (err) {
        alert(`Pipeline failure: ${err}`);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `<i data-lucide="play-circle" class="w-4 h-4"></i><span>Execute Multi-Step Pipeline</span>`;
        if (window.lucide) lucide.createIcons();
    }
}

function renderArtifactTabs(workflowData) {
    const container = document.getElementById("artifactTabsContainer");
    container.innerHTML = "";

    // 1. Final synthesis
    const finalBtn = document.createElement("button");
    finalBtn.className = "artifact-tab active";
    finalBtn.innerText = "Final Synthesis";
    finalBtn.onclick = () => {
        setActiveArtifactTab(finalBtn);
        displayPipelineArtifact("final");
    };
    container.appendChild(finalBtn);

    // 2. Intermediate artifacts
    const artifacts = workflowData.intermediate_artifacts || {};
    Object.keys(artifacts).forEach(k => {
        const btn = document.createElement("button");
        btn.className = "artifact-tab";
        btn.innerText = k.replace(/_/g, " ").toUpperCase();
        btn.onclick = () => {
            setActiveArtifactTab(btn);
            displayPipelineArtifact(k);
        };
        container.appendChild(btn);
    });
}

function setActiveArtifactTab(activeBtn) {
    document.querySelectorAll(".artifact-tab").forEach(b => b.classList.remove("active"));
    activeBtn.classList.add("active");
}

function displayPipelineArtifact(key) {
    const viewer = document.getElementById("artifactViewer");
    if (!appState.activePipelineContext) return;

    if (key === "final") {
        viewer.innerText = appState.activePipelineContext.final_output;
    } else {
        const item = appState.activePipelineContext.intermediate_artifacts[key];
        if (typeof item === "object") {
            viewer.innerText = JSON.stringify(item, null, 2);
        } else {
            viewer.innerText = item;
        }
    }
}


// =====================================================================
// Validation & Guardrails
// =====================================================================

async function runValidationOnCurrentOutput() {
    if (!appState.currentOutput) {
        alert("No generated output available. Generate text first!");
        return;
    }

    try {
        showLoading(true, "Verifying schema conformance and factual grounding...");
        const res = await fetch("/api/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                output_text: appState.currentOutput,
            }),
        });
        const data = await res.json();
        if (data.success) {
            appState.lastValidation = data.validation;
            renderValidationResults(data.validation);
            switchTab("structured_validation");
        }
    } catch (err) {
        alert(`Validation error: ${err}`);
    } finally {
        showLoading(false);
    }
}

function renderValidationResults(val) {
    const pill = document.getElementById("valStatusPill");
    if (val.is_valid) {
        pill.className = "inline-flex items-center space-x-2 px-4 py-2 rounded-full text-base font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
        pill.innerText = "Valid & Grounded";
    } else {
        pill.className = "inline-flex items-center space-x-2 px-4 py-2 rounded-full text-base font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20";
        pill.innerText = "Validation Failed";
    }

    document.getElementById("valQualityScore").innerText = `${val.quality_score}%`;
    document.getElementById("valGroundingRatio").innerText = `${Math.round(val.grounding.grounding_ratio * 100)}%`;
    document.getElementById("valRepairedFlag").innerText = val.was_repaired ? "Yes" : "No";

    // Diagnostics list
    const diagList = document.getElementById("valDiagnosticList");
    diagList.innerHTML = "";

    const allItems = [...val.errors, ...val.warnings];
    if (allItems.length === 0) {
        diagList.innerHTML = `<div class="p-3 bg-emerald-950/30 border border-emerald-500/20 text-emerald-300 rounded-lg">Zero validation errors. All schema integrity and safety guardrails passed!</div>`;
    } else {
        allItems.forEach(item => {
            const isError = item.severity === "ERROR";
            const row = document.createElement("div");
            row.className = `p-3 rounded-lg border ${isError ? "bg-rose-950/30 border-rose-500/30 text-rose-300" : "bg-amber-950/30 border-amber-500/30 text-amber-300"}`;
            row.innerHTML = `<strong>[${item.code}]</strong> ${escapeHtml(item.message)}`;
            diagList.appendChild(row);
        });
    }

    // Grounding items breakdown
    document.getElementById("valFactsChecked").innerText = val.grounding.total_facts_checked;
    document.getElementById("valFactsVerified").innerText = val.grounding.verified_facts;
    document.getElementById("valFactsUnverified").innerText = val.grounding.unverified_facts;

    const gList = document.getElementById("groundingItemsList");
    gList.innerHTML = "";
    val.grounding.verified_items.slice(0, 6).forEach(v => {
        gList.innerHTML += `<div class="text-emerald-400 font-mono">✓ Verified in Source: ${escapeHtml(v)}</div>`;
    });
    val.grounding.unverified_items.slice(0, 4).forEach(u => {
        gList.innerHTML += `<div class="text-rose-400 font-mono">⚠ Ungrounded: ${escapeHtml(u)}</div>`;
    });

    if (val.parsed_json) {
        document.getElementById("structuredJsonViewer").innerText = JSON.stringify(val.parsed_json, null, 2);
    }
}

async function generateStructuredPydanticOutput() {
    if (!appState.currentDocument) {
        alert("Please ingest a document first.");
        return;
    }

    try {
        showLoading(true, "Generating schema-enforced structured JSON...");
        const res = await fetch("/api/prompt/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                template_id: "struct_pydantic_eval",
                requirements: "Extract all core intelligence conforming strictly to schema.",
                output_format: "json",
            }),
        });
        const data = await res.json();
        if (data.success) {
            appState.currentOutput = data.response.content;
            document.getElementById("structuredJsonViewer").innerText = data.response.content;
            runValidationOnCurrentOutput();
        }
    } catch (err) {
        alert(`Schema generation error: ${err}`);
    } finally {
        showLoading(false);
    }
}


// =====================================================================
// Multi-Dimensional Evaluation & Radar Benchmarks
// =====================================================================

function initRadarChart() {
    const ctx = document.getElementById("evalRadarChart");
    if (!ctx) return;

    appState.radarChart = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["Factuality", "Relevance", "Consistency", "Completeness", "Readability"],
            datasets: [{
                label: "Quality Score",
                data: [85, 85, 80, 85, 75],
                backgroundColor: "rgba(20, 184, 166, 0.25)",
                borderColor: "rgba(45, 212, 191, 1)",
                borderWidth: 2.5,
                pointBackgroundColor: "rgba(52, 211, 153, 1)",
                pointBorderColor: "#ffffff",
                pointHoverBackgroundColor: "#ffffff",
                pointHoverBorderColor: "rgba(20, 184, 166, 1)",
                pointRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "rgba(35, 55, 85, 0.7)" },
                    grid: { color: "rgba(35, 55, 85, 0.5)" },
                    pointLabels: {
                        color: "#94a3b8",
                        font: { size: 11, family: "sans-serif", weight: "600" }
                    },
                    ticks: {
                        backdropColor: "transparent",
                        color: "#64748b",
                        stepSize: 20,
                        min: 0,
                        max: 100
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function runEvaluationOnCurrentOutput() {
    if (!appState.currentOutput) {
        alert("Please generate or synthesize content before evaluating.");
        return;
    }

    try {
        showLoading(true, "Executing multi-criteria evaluation engine...");
        const res = await fetch("/api/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                output_text: appState.currentOutput,
                requirements: document.getElementById("promptReqInput")?.value || "",
            }),
        });
        const data = await res.json();
        if (data.success) {
            appState.lastEvaluation = data.evaluation;
            renderEvaluationReport(data.evaluation);
            switchTab("evaluation");
        }
    } catch (err) {
        alert(`Evaluation error: ${err}`);
    } finally {
        showLoading(false);
    }
}

function renderEvaluationReport(ev) {
    document.getElementById("evalGradeBadge").innerText = `Grade: ${ev.grade}`;
    document.getElementById("evalOverallIndex").innerText = Math.round(ev.overall_quality_index);

    // Dimension cards
    document.getElementById("evalFactualityPct").innerText = `${ev.factuality.percentage}%`;
    document.getElementById("evalFactualityBar").style.width = `${ev.factuality.percentage}%`;
    document.getElementById("evalFactualityExpl").innerText = ev.factuality.explanation;

    document.getElementById("evalRelevancePct").innerText = `${ev.relevance.percentage}%`;
    document.getElementById("evalRelevanceBar").style.width = `${ev.relevance.percentage}%`;
    document.getElementById("evalRelevanceExpl").innerText = ev.relevance.explanation;

    document.getElementById("evalConsistencyPct").innerText = `${ev.consistency.percentage}%`;
    document.getElementById("evalConsistencyBar").style.width = `${ev.consistency.percentage}%`;
    document.getElementById("evalConsistencyExpl").innerText = ev.consistency.explanation;

    document.getElementById("evalReadabilityPct").innerText = `${ev.readability.percentage}%`;
    document.getElementById("evalReadabilityBar").style.width = `${ev.readability.percentage}%`;
    document.getElementById("evalReadabilityExpl").innerText = ev.readability.explanation;

    // ROUGE Scores
    document.getElementById("rouge1F1").innerText = ev.rouge.rouge_1_f1;
    document.getElementById("rouge1Prec").innerText = ev.rouge.rouge_1_precision;
    document.getElementById("rouge1Rec").innerText = ev.rouge.rouge_1_recall;

    document.getElementById("rouge2F1").innerText = ev.rouge.rouge_2_f1;
    document.getElementById("rouge2Prec").innerText = ev.rouge.rouge_2_precision;
    document.getElementById("rouge2Rec").innerText = ev.rouge.rouge_2_recall;

    document.getElementById("rougeLF1").innerText = ev.rouge.rouge_l_f1;
    document.getElementById("rougeLPrec").innerText = ev.rouge.rouge_l_precision;
    document.getElementById("rougeLRec").innerText = ev.rouge.rouge_l_recall;

    // Recommendations
    const recList = document.getElementById("evalRecommendationsList");
    recList.innerHTML = "";
    ev.recommendations.forEach(r => {
        recList.innerHTML += `<li class="flex items-start space-x-2"><span class="text-amber-400">•</span><span>${escapeHtml(r)}</span></li>`;
    });

    // Update Radar Chart
    if (appState.radarChart && ev.radar_data) {
        appState.radarChart.data.datasets[0].data = [
            ev.radar_data.Factuality,
            ev.radar_data.Relevance,
            ev.radar_data.Consistency,
            ev.radar_data.Completeness,
            ev.radar_data.Readability,
        ];
        appState.radarChart.update();
    }
}


// =====================================================================
// Settings & Export Helpers
// =====================================================================

async function saveSettings() {
    const key = document.getElementById("settingsApiKeyInput").value;
    const model = document.getElementById("settingsModelSelect").value;
    const mode = document.getElementById("settingsEngineModeSelect").value;

    try {
        const res = await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                api_key: key || undefined,
                model: model,
                mode: mode,
            }),
        });
        const data = await res.json();
        if (data.success) {
            alert("Settings updated successfully!");
            fetchStatus();
        }
    } catch (err) {
        alert(`Failed to save settings: ${err}`);
    }
}

function copyOutput() {
    if (!appState.currentOutput) return;
    navigator.clipboard.writeText(appState.currentOutput).then(() => {
        alert("Copied generated text to clipboard!");
    });
}

function downloadOutput(filename = "document_intelligence_output.md") {
    if (!appState.currentOutput) return;
    const blob = new Blob([appState.currentOutput], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function showLoading(isLoading, message = "Processing...") {
    let loader = document.getElementById("globalLoaderOverlay");
    if (isLoading) {
        if (!loader) {
            loader = document.createElement("div");
            loader.id = "globalLoaderOverlay";
            loader.className = "fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center space-y-4";
            loader.innerHTML = `
                <div class="h-12 w-12 rounded-full border-4 border-teal-400 border-t-transparent animate-spin"></div>
                <div id="globalLoaderMsg" class="text-sm font-semibold text-white tracking-wide"></div>
            `;
            document.body.appendChild(loader);
        }
        document.getElementById("globalLoaderMsg").innerText = message;
        loader.classList.remove("hidden");
    } else if (loader) {
        loader.classList.add("hidden");
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
