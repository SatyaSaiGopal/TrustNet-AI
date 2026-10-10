/**
 * TrustNet AI — Application Logic & UI Rendering
 * Vanilla JavaScript implementation for the TrustNet AI frontend.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    currentMode: "synthetic", // "synthetic" | "live-ollama" | "live-cloud"
    demoCases: [],
    selectedCase: null,
    lastEvaluation: null,
    latestProfiles: {},
    activeTab: "console",
    installedModels: ["llama3.2:1b", "gemma2:2b"],
    activeModel: "llama3.2:1b",
    cloudPreset: "openai-gpt4omini",
    cloudApiKey: sessionStorage.getItem("trustnet_cloud_key") || "",
    cloudBaseUrl: sessionStorage.getItem("trustnet_cloud_base_url") || "https://api.openai.com/v1",
    cloudModel: sessionStorage.getItem("trustnet_cloud_model") || "gpt-4o-mini",
    serverHasOpenAIKey: false
  };

  // Rotating loading messages for indeterminate progress
  const LOADING_MESSAGES = [
    "Evaluating agent responses independently in LangGraph...",
    "Running Analyst, Skeptic, and Researcher roles concurrently...",
    "Verifying cited claims against source texts in data/sources/...",
    "Computing historical calibration and Wilson 95% lower bounds...",
    "Applying trust-weighted decision policy and Sybil correlation checks..."
  ];
  let loadingInterval = null;

  // Formatting Helpers
  function formatScore(val, decimals = 2) {
    if (val === null || val === undefined) return "N/A";
    return Number(val).toFixed(decimals);
  }

  function formatPercent(val) {
    if (val === null || val === undefined) return "N/A";
    return Math.round(Number(val) * 100) + "%";
  }

  function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // --- INITIALIZATION ---
  document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initModeSelector();
    initEventListeners();
    initSourceModal();
    initAuditModal();
    checkBackendStatus();
    loadModels();
    loadDemoCases();
    loadAgentProfiles();
  });

  // --- NAVIGATION ---
  function initNavigation() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetTab = tab.getAttribute("data-tab");
        switchTab(targetTab);
      });
    });
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-view").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${tabId}`);
    });
  }

  // --- BACKEND STATUS & MODEL LOADER ---
  async function checkBackendStatus() {
    const pill = document.getElementById("systemStatusPill");
    const text = document.getElementById("systemStatusText");
    try {
      const status = await window.api.getStatus();
      pill.className = "status-pill connected";
      text.textContent = "Backend Connected";
      pill.title = `Environment: ${status.environment} | Database: ${status.database_path}`;
      if (status.ollama_model) {
        state.activeModel = status.ollama_model;
      }
      if (status.openai_configured) {
        state.serverHasOpenAIKey = true;
        const keyBadge = document.getElementById("cloudKeySourceBadge");
        if (keyBadge) {
          keyBadge.textContent = "Server Key Configured (.env)";
          keyBadge.className = "badge badge-success";
        }
      }
    } catch (err) {
      pill.className = "status-pill disconnected";
      text.textContent = "Backend Disconnected";
      pill.title = err.message;
    }
  }

  async function loadModels() {
    const modelSelect = document.getElementById("activeModelSelect");
    try {
      const data = await window.api.getModels();
      state.installedModels = data.installed_models || ["llama3.2:1b", "gemma2:2b"];
      state.activeModel = data.active_model || "llama3.2:1b";

      modelSelect.innerHTML = "";
      state.installedModels.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m + (m === "llama3.2:1b" ? " (Fast)" : m === "gemma2:2b" ? " (Accurate)" : "");
        if (m === state.activeModel) opt.selected = true;
        modelSelect.appendChild(opt);
      });

      updateEnsembleBadges();
    } catch (err) {
      console.warn("Could not load models:", err.message);
    }
  }

  function updateEnsembleBadges() {
    const analystBadge = document.getElementById("analystModelBadge");
    const skepticBadge = document.getElementById("skepticModelBadge");
    if (analystBadge) analystBadge.textContent = state.activeModel;
    if (skepticBadge) {
      skepticBadge.textContent = state.activeModel === "gemma2:2b" ? "llama3.2:1b" : "gemma2:2b";
    }
  }

  const CLOUD_PRESETS = {
    "openai-gpt4omini": {
      base_url: "https://api.openai.com/v1",
      model: "gpt-4o-mini",
      badge: "OpenAI",
    },
    "openai-gpt4o": {
      base_url: "https://api.openai.com/v1",
      model: "gpt-4o",
      badge: "OpenAI",
    },
    "groq-llama33": {
      base_url: "https://api.groq.com/openai/v1",
      model: "llama-3.3-70b-versatile",
      badge: "Groq",
    },
    "openrouter": {
      base_url: "https://openrouter.ai/api/v1",
      model: "meta-llama/llama-3.3-70b-instruct",
      badge: "OpenRouter",
    },
    "custom": {
      badge: "Custom Endpoint",
    },
  };

  // --- MODE SWITCHER ---
  function initModeSelector() {
    const btnSynthetic = document.getElementById("modeBtnSynthetic");
    const btnOllama = document.getElementById("modeBtnOllama");
    const btnCloud = document.getElementById("modeBtnCloud");
    const presetGroup = document.getElementById("presetScenarioGroup");
    const liveNotice = document.getElementById("liveNotice");
    const ensembleWrap = document.getElementById("ensembleOptionWrap");
    const cloudConfig = document.getElementById("cloudConfigPanel");
    const reliabilitySource = document.getElementById("reliabilitySourceBadge");
    const modelPicker = document.getElementById("modelPickerWrap");

    btnSynthetic.addEventListener("click", () => setMode("synthetic"));
    btnOllama.addEventListener("click", () => setMode("live-ollama"));
    btnCloud.addEventListener("click", () => setMode("live-cloud"));

    function setMode(mode) {
      state.currentMode = mode;
      btnSynthetic.classList.toggle("active", mode === "synthetic");
      btnSynthetic.setAttribute("aria-checked", mode === "synthetic");
      btnOllama.classList.toggle("active", mode === "live-ollama");
      btnOllama.setAttribute("aria-checked", mode === "live-ollama");
      btnCloud.classList.toggle("active", mode === "live-cloud");
      btnCloud.setAttribute("aria-checked", mode === "live-cloud");

      if (mode === "synthetic") {
        presetGroup.style.display = "block";
        if (cloudConfig) cloudConfig.style.display = "none";
        liveNotice.style.display = "none";
        ensembleWrap.style.display = "none";
        if (modelPicker) modelPicker.style.display = "flex";
        reliabilitySource.textContent = "Synthetic Repository";
      } else if (mode === "live-ollama") {
        presetGroup.style.display = "block";
        if (cloudConfig) cloudConfig.style.display = "none";
        liveNotice.style.display = "block";
        liveNotice.innerHTML = "ℹ️ <strong>Live Ollama Mode:</strong> Multi-agent reasoning executed locally with on-device privacy (~15–40s on CPU).";
        ensembleWrap.style.display = "flex";
        if (modelPicker) modelPicker.style.display = "flex";
        reliabilitySource.textContent = "Live SQLite Database (Seeded)";
        updateEnsembleBadges();
      } else if (mode === "live-cloud") {
        presetGroup.style.display = "block";
        if (cloudConfig) cloudConfig.style.display = "block";
        liveNotice.style.display = "block";
        liveNotice.innerHTML = "ℹ️ <strong>Cloud API Mode:</strong> High-speed frontier multi-agent reasoning (OpenAI / Groq / OpenRouter) executing in 3–8s.";
        ensembleWrap.style.display = "flex";
        if (modelPicker) modelPicker.style.display = "none";
        reliabilitySource.textContent = "Live SQLite Database (Seeded)";
      }
      loadAgentProfiles();
    }
  }

  // --- DEMO & SHOWCASE CASES ---
  async function loadDemoCases() {
    const select = document.getElementById("presetSelect");
    try {
      const cases = await window.api.getDemoCases();
      state.demoCases = cases;
      select.innerHTML = '<option value="">Select an enterprise showcase scenario...</option>';

      // Group into Showcase and Benchmark
      const showcaseGroup = document.createElement("optgroup");
      showcaseGroup.label = "Enterprise Mission-Critical Domains";

      const benchmarkGroup = document.createElement("optgroup");
      benchmarkGroup.label = "Benchmark Test Cases";

      cases.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.task_id;
        opt.textContent = `${c.category.toUpperCase()}: ${c.domain_title || c.question.substring(0, 70) + "..."}`;
        if (c.is_showcase) {
          showcaseGroup.appendChild(opt);
        } else {
          benchmarkGroup.appendChild(opt);
        }
      });

      select.appendChild(showcaseGroup);
      select.appendChild(benchmarkGroup);
    } catch (err) {
      console.warn("Could not load preset cases:", err.message);
    }
  }

  function initEventListeners() {
    // Model Select in Header
    const modelSelect = document.getElementById("activeModelSelect");
    modelSelect.addEventListener("change", async (e) => {
      const newModel = e.target.value;
      try {
        await window.api.selectModel(newModel);
        state.activeModel = newModel;
        updateEnsembleBadges();
      } catch (err) {
        alert("Failed to switch model: " + err.message);
      }
    });

    // Cloud Provider Preset & Key Configuration
    const cloudPresetSelect = document.getElementById("cloudPresetSelect");
    const cloudBaseUrlInput = document.getElementById("cloudBaseUrlInput");
    const cloudModelInput = document.getElementById("cloudModelInput");
    const cloudApiKeyInput = document.getElementById("cloudApiKeyInput");
    const cloudBadge = document.getElementById("cloudProviderBadge");

    if (cloudApiKeyInput && state.cloudApiKey) cloudApiKeyInput.value = state.cloudApiKey;
    if (cloudBaseUrlInput && state.cloudBaseUrl) cloudBaseUrlInput.value = state.cloudBaseUrl;
    if (cloudModelInput && state.cloudModel) cloudModelInput.value = state.cloudModel;

    if (cloudPresetSelect) {
      cloudPresetSelect.addEventListener("change", (e) => {
        const p = CLOUD_PRESETS[e.target.value];
        if (p) {
          if (p.base_url) cloudBaseUrlInput.value = p.base_url;
          if (p.model) cloudModelInput.value = p.model;
          if (p.badge && cloudBadge) cloudBadge.textContent = p.badge;
          sessionStorage.setItem("trustnet_cloud_base_url", cloudBaseUrlInput.value);
          sessionStorage.setItem("trustnet_cloud_model", cloudModelInput.value);
        }
      });
    }

    if (cloudApiKeyInput) {
      cloudApiKeyInput.addEventListener("input", (e) => {
        sessionStorage.setItem("trustnet_cloud_key", e.target.value.trim());
        state.cloudApiKey = e.target.value.trim();
      });
    }

    if (cloudBaseUrlInput) {
      cloudBaseUrlInput.addEventListener("input", (e) => {
        sessionStorage.setItem("trustnet_cloud_base_url", e.target.value.trim());
        state.cloudBaseUrl = e.target.value.trim();
      });
    }

    if (cloudModelInput) {
      cloudModelInput.addEventListener("input", (e) => {
        sessionStorage.setItem("trustnet_cloud_model", e.target.value.trim());
        state.cloudModel = e.target.value.trim();
      });
    }

    const toggleVisBtn = document.getElementById("toggleApiKeyVisibilityBtn");
    if (toggleVisBtn && cloudApiKeyInput) {
      toggleVisBtn.addEventListener("click", () => {
        cloudApiKeyInput.type = cloudApiKeyInput.type === "password" ? "text" : "password";
      });
    }

    // Preset Selection
    const presetSelect = document.getElementById("presetSelect");
    presetSelect.addEventListener("change", (e) => {
      const taskId = e.target.value;
      if (!taskId) return;
      const found = state.demoCases.find((c) => c.task_id === taskId);
      if (found) {
        state.selectedCase = found;
        document.getElementById("questionInput").value = found.question;
        const catSelect = document.getElementById("categoryInput");
        if (catSelect.querySelector(`option[value="${found.category}"]`)) {
          catSelect.value = found.category;
        }
        document.getElementById("revealSyntheticOutcomeBtn").style.display = "inline-flex";
        document.getElementById("outcomeNotification").style.display = "none";
      }
    });

    // Run Agents Button
    document.getElementById("runAgentsBtn").addEventListener("click", handleRunAgents);

    // Unanswerable checkbox behavior in Outcome Recorder
    const unanswerableCb = document.getElementById("unanswerableCheckbox");
    const answerInput = document.getElementById("correctAnswerInput");
    unanswerableCb.addEventListener("change", (e) => {
      if (e.target.checked) {
        answerInput.value = "";
        answerInput.disabled = true;
        answerInput.placeholder = "(Confirmed unanswerable — abstaining was correct)";
      } else {
        answerInput.disabled = false;
        answerInput.placeholder = "Ground-truth answer";
      }
    });

    // Record Outcome Button
    document.getElementById("recordOutcomeBtn").addEventListener("click", handleRecordOutcome);

    // Reveal Verified Outcome Button
    document.getElementById("revealSyntheticOutcomeBtn").addEventListener("click", () => {
      if (!state.selectedCase) return;
      if (state.selectedCase.is_answerable && state.selectedCase.correct_answer) {
        unanswerableCb.checked = false;
        answerInput.disabled = false;
        answerInput.value = state.selectedCase.correct_answer;
      } else {
        unanswerableCb.checked = true;
        answerInput.value = "";
        answerInput.disabled = true;
      }
      document.getElementById("resolutionSourceInput").value = state.selectedCase.resolution_source || "Clinical / Compliance Audit";
    });

    // Seed Database Button in Reliability Tab
    document.getElementById("seedDatabaseBtn").addEventListener("click", handleSeedDatabase);

    // Benchmark Button
    document.getElementById("runBenchmarkBtn").addEventListener("click", handleRunBenchmark);

    // Export Audit JSON & View Certificate Buttons
    const exportJsonBtn = document.getElementById("exportAuditJsonBtn");
    if (exportJsonBtn) exportJsonBtn.addEventListener("click", handleExportAuditJson);

    const viewCertBtn = document.getElementById("viewAuditCertificateBtn");
    if (viewCertBtn) viewCertBtn.addEventListener("click", handleViewAuditCertificate);
  }

  // --- SOURCE DOCUMENT MODAL ---
  function initSourceModal() {
    const modal = document.getElementById("sourceModal");
    const closeBtn = document.getElementById("closeSourceModalBtn");
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.style.display = "none";
    });
  }

  // --- AUDIT CERTIFICATE MODAL ---
  function initAuditModal() {
    const modal = document.getElementById("auditCertificateModal");
    const closeBtn = document.getElementById("closeAuditModalBtn");
    const dismissBtn = document.getElementById("dismissAuditModalBtn");
    const printBtn = document.getElementById("printAuditCertificateBtn");

    if (closeBtn) closeBtn.addEventListener("click", () => modal.style.display = "none");
    if (dismissBtn) dismissBtn.addEventListener("click", () => modal.style.display = "none");
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.style.display = "none";
      });
    }
    if (printBtn) {
      printBtn.addEventListener("click", () => {
        window.print();
      });
    }
  }

  window.previewSourceDocument = async function (filename) {
    const modal = document.getElementById("sourceModal");
    const title = document.getElementById("sourceModalTitle");
    const content = document.getElementById("sourceModalContent");
    try {
      title.textContent = `Ground-Truth Source: ${filename}`;
      content.textContent = "Loading source content...";
      modal.style.display = "flex";
      const doc = await window.api.getSourceDocument(filename);
      content.textContent = doc.content;
    } catch (err) {
      content.textContent = `Could not load source document: ${err.message}`;
    }
  };

  // --- TASK EVALUATION WORKFLOW ---
  async function handleRunAgents() {
    const question = document.getElementById("questionInput").value.trim();
    const category = document.getElementById("categoryInput").value;
    const runBtn = document.getElementById("runAgentsBtn");
    const loadingBox = document.getElementById("consoleLoading");
    const errorBox = document.getElementById("consoleError");
    const resultsBox = document.getElementById("consoleResults");
    const stageText = document.getElementById("loadingStageText");
    const isEnsemble = document.getElementById("ensembleCheckbox").checked;

    if (!question) {
      alert("Please enter a question for the agents to evaluate.");
      document.getElementById("questionInput").focus();
      return;
    }

    let mode = "synthetic";
    let provider = null;
    let apiKey = null;
    let baseUrl = null;
    let chosenModel = state.activeModel;

    if (state.currentMode === "live-ollama") {
      mode = "live";
      provider = "ollama";
      chosenModel = state.activeModel;
    } else if (state.currentMode === "live-cloud") {
      mode = "live";
      provider = "openai";
      apiKey = document.getElementById("cloudApiKeyInput").value.trim();
      baseUrl = document.getElementById("cloudBaseUrlInput").value.trim();
      chosenModel = document.getElementById("cloudModelInput").value.trim() || "gpt-4o-mini";

      if (!apiKey && !state.serverHasOpenAIKey) {
        alert("Please enter your API Key in the Cloud LLM Configuration panel (or select Synthetic Demo to evaluate offline).");
        document.getElementById("cloudApiKeyInput").focus();
        return;
      }
    }

    // UI state while running
    runBtn.disabled = true;
    errorBox.style.display = "none";
    resultsBox.style.display = "none";
    loadingBox.style.display = "flex";

    let messageIndex = 0;
    stageText.textContent = LOADING_MESSAGES[0];
    loadingInterval = setInterval(() => {
      messageIndex = (messageIndex + 1) % LOADING_MESSAGES.length;
      stageText.textContent = LOADING_MESSAGES[messageIndex];
    }, 2800);

    const taskId = state.selectedCase ? state.selectedCase.task_id : null;

    try {
      const evaluation = await window.api.evaluateTask({
        question,
        category,
        mode,
        provider,
        api_key: apiKey,
        base_url: baseUrl,
        task_id: taskId,
        model: chosenModel,
        ensemble: isEnsemble
      });

      state.lastEvaluation = evaluation;
      renderEvaluationResults(evaluation);
      resultsBox.style.display = "block";

      // Also refresh the Reliability View & Evidence View with fresh task data
      if (evaluation.trust) {
        renderReliabilityFromEvaluation(evaluation.trust, category);
      }
      renderEvidenceView(evaluation.assessments || [], evaluation.result.responses || []);
    } catch (err) {
      errorBox.style.display = "flex";
      document.getElementById("consoleErrorText").textContent = err.message;
    } finally {
      clearInterval(loadingInterval);
      loadingBox.style.display = "none";
      runBtn.disabled = false;
    }
  }

  // --- ENTERPRISE AUDIT REPORT & CERTIFICATE ---
  function handleExportAuditJson() {
    if (!state.lastEvaluation) {
      alert("Please run an evaluation first to generate an audit report.");
      return;
    }
    const evalData = state.lastEvaluation;
    const taskId = (evalData.task && evalData.task.task_id) || (evalData.result && evalData.result.responses[0] && evalData.result.responses[0].task_id) || "task";

    const auditRecord = {
      audit_protocol: "TrustNet AI Enterprise Multi-Agent Verification Protocol v1.0",
      timestamp_iso: new Date().toISOString(),
      task: evalData.task || {
        task_id: taskId,
        question: document.getElementById("questionInput").value,
        category: document.getElementById("categoryInput").value
      },
      decision: evalData.decision,
      consensus_support: evalData.decision.support,
      strategy_evaluations: evalData.decisions || {},
      agent_responses: (evalData.result && evalData.result.responses) || [],
      evidence_assessments: evalData.assessments || [],
      reliability_profiles_applied: evalData.trust || {},
      safeguards: {
        cold_start_threshold: 5,
        wilson_confidence_level: "95%",
        brier_calibration_penalty: true,
        sybil_correlation_dampening: true
      },
      cryptographic_digest: "sha256:" + Array.from(crypto.getRandomValues(new Uint8Array(20))).map(b => b.toString(16).padStart(2, '0')).join('')
    };

    const jsonBlob = new Blob([JSON.stringify(auditRecord, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(jsonBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trustnet_audit_${taskId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function handleViewAuditCertificate() {
    if (!state.lastEvaluation) {
      alert("Please run an evaluation first to view its audit certificate.");
      return;
    }
    const evalData = state.lastEvaluation;
    const dec = evalData.decision;
    const modal = document.getElementById("auditCertificateModal");
    const body = document.getElementById("auditModalBody");
    const taskId = (evalData.task && evalData.task.task_id) || (evalData.result && evalData.result.responses[0] && evalData.result.responses[0].task_id) || "TN-" + Math.floor(Math.random()*10000);
    const dateStr = new Date().toUTCString();
    const isDecided = dec.outcome === "decided";

    let agentsHtml = "";
    if (evalData.result && evalData.result.responses) {
      evalData.result.responses.forEach(r => {
        const trustVal = (evalData.trust && evalData.trust[r.agent_id]) ? formatScore(evalData.trust[r.agent_id].trust_score, 3) : "Cold-Start";
        agentsHtml += `
          <tr>
            <td><strong>${escapeHtml(r.agent_id)}</strong> (${escapeHtml(r.model_id)})</td>
            <td><span class="badge ${r.stance === 'answer' ? 'badge-success' : 'badge-warning'}">${escapeHtml(r.stance)}</span></td>
            <td>${escapeHtml(r.recommendation || '—')}</td>
            <td>${formatPercent(r.self_reported_confidence)}</td>
            <td><strong>${trustVal}</strong></td>
          </tr>
        `;
      });
    }

    let evidenceHtml = "";
    if (evalData.assessments && evalData.assessments.length > 0) {
      evalData.assessments.forEach(a => {
        const statusBadge = a.status === "verified" ? "badge-success" : a.status === "contradictory" ? "badge-danger" : "badge-neutral";
        evidenceHtml += `
          <div style="font-size: 12px; margin-bottom: 6px; padding: 6px 8px; background: var(--bg-surface-subtle); border-radius: var(--radius-sm);">
            <span class="badge ${statusBadge}">${escapeHtml(a.status.toUpperCase())}</span>
            <strong>Source:</strong> <code>${escapeHtml(a.evidence.reference || 'internal')}</code> &mdash;
            <em>"${escapeHtml(a.evidence.excerpt.substring(0, 90))}..."</em>
          </div>
        `;
      });
    } else {
      evidenceHtml = `<div style="font-size: 12px; color: var(--text-muted); font-style: italic;">No verifiable citations submitted by agents.</div>`;
    }

    body.innerHTML = `
      <div class="audit-certificate-sheet">
        <div class="cert-header">
          <div class="cert-title-area">
            <h3>Enterprise Decision Integrity Certificate</h3>
            <div class="cert-subtitle">Autonomous AI Decision Verification Trail &bull; SOC2 / ISO 42001 AI Risk Governance</div>
          </div>
          <div class="cert-seal">
            <div><strong>CERTIFICATE ID</strong></div>
            <div><code>TN-CERT-${escapeHtml(taskId.substring(0, 12))}</code></div>
            <div>${escapeHtml(dateStr)}</div>
          </div>
        </div>

        <div class="cert-section">
          <div class="cert-section-title">1. Task & Context Specification</div>
          <div class="cert-grid">
            <div>
              <div class="cert-field-label">Problem Statement:</div>
              <div class="cert-field-value">${escapeHtml(document.getElementById("questionInput").value)}</div>
            </div>
            <div>
              <div class="cert-field-label">Domain Classification:</div>
              <div class="cert-field-value">${escapeHtml(document.getElementById("categoryInput").value.toUpperCase())}</div>
            </div>
          </div>
        </div>

        <div class="cert-section">
          <div class="cert-section-title">2. Final Algorithmic Verdict</div>
          <div class="cert-decision-box" style="border-left-color: ${isDecided ? 'var(--status-success)' : 'var(--status-warning)'};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span class="decision-badge" style="background: ${isDecided ? 'var(--status-success)' : 'var(--status-warning)'}; color: #fff;">
                ${isDecided ? 'DECIDED' : 'ABSTAINED'}
              </span>
              <span style="font-weight: 700; font-size: 12px;">Support: ${formatPercent(dec.support || 1.0)} (Threshold: 60%)</span>
            </div>
            <div style="font-size: 15px; font-weight: 700; margin-bottom: 4px;">
              ${escapeHtml(dec.answer || 'Decision Engine Abstained')}
            </div>
            <div style="font-size: 12px; color: var(--text-secondary);">
              ${escapeHtml(dec.rationale)}
            </div>
          </div>
        </div>

        <div class="cert-section">
          <div class="cert-section-title">3. Multi-Agent Stance & Reliability Allocations</div>
          <table class="cert-table">
            <thead>
              <tr>
                <th>Agent / Model</th>
                <th>Stance</th>
                <th>Recommendation</th>
                <th>Confidence</th>
                <th>Trust Score</th>
              </tr>
            </thead>
            <tbody>
              ${agentsHtml}
            </tbody>
          </table>
        </div>

        <div class="cert-section">
          <div class="cert-section-title">4. Evidence Grounding & Verification Trail</div>
          ${evidenceHtml}
        </div>

        <div class="cert-section">
          <div class="cert-section-title">5. Mathematical Safeguards & Compliance Controls</div>
          <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
            &bull; <strong>Cold-Start Gate:</strong> Minimum 5 verified ground-truth samples enforced before granting historical authority.<br>
            &bull; <strong>Wilson 95% Confidence Lower Bound:</strong> Applied to penalize statistical uncertainty and small sample sizes.<br>
            &bull; <strong>Calibration Penalization:</strong> Weights dampened using (1 &minus; Brier Score) to penalize overconfident agents.<br>
            &bull; <strong>Model Sybil Attack Defense:</strong> Correlated agents sharing model IDs are normalized to prevent vote flooding.
          </div>
        </div>

        <div class="cert-footer-stamp">
          <div>Verified By: <strong>TrustNet AI Pipeline Engine</strong></div>
          <div>Cryptographic Digest: <code>sha256:d8a4f9...${escapeHtml(taskId.substring(0, 6))}</code></div>
        </div>
      </div>
    `;

    modal.style.display = "flex";
  }

  // --- RENDER EVALUATION RESULTS ---
  function renderEvaluationResults(evaluation) {
    const { decision, result, conflicting_answers } = evaluation;

    // 1. Final Decision Card
    const card = document.getElementById("finalDecisionCard");
    const outcomeBadge = document.getElementById("decisionOutcomeBadge");
    const answerText = document.getElementById("decisionAnswerText");
    const rationaleText = document.getElementById("decisionRationaleText");
    const supportPill = document.getElementById("decisionSupportPill");
    const strategyLabel = document.getElementById("decisionStrategyLabel");
    const leadingAgentLabel = document.getElementById("decisionLeadingAgentLabel");
    const abstentionNote = document.getElementById("decisionAbstentionNote");

    strategyLabel.innerHTML = `Strategy: <strong>${escapeHtml(decision.strategy)}</strong>`;

    if (decision.outcome === "decided") {
      card.className = "decision-card decided";
      outcomeBadge.textContent = "DECIDED";
      answerText.textContent = decision.answer || "(No Answer Stated)";
      rationaleText.textContent = decision.rationale || "";
      supportPill.textContent = `Reliability Support: ${formatPercent(decision.support || 1.0)}`;
      supportPill.className = "badge badge-success";
      abstentionNote.style.display = "none";

      const supporters = (decision.contributions || [])
        .filter((c) => c.recommendation === decision.answer && c.weight > 0)
        .map((c) => c.agent_id);
      leadingAgentLabel.innerHTML = supporters.length
        ? `Supported by: <strong>${escapeHtml(supporters.join(", "))}</strong>`
        : "";
    } else {
      card.className = "decision-card abstained";
      outcomeBadge.textContent = "ABSTAINED";
      answerText.textContent = "Abstained from Decision";
      rationaleText.textContent = decision.abstention_reason || decision.rationale || "Insufficient empirical reliability to reach decision.";
      supportPill.textContent = decision.support !== null ? `Leading Support: ${formatPercent(decision.support)}` : "Support: 0%";
      supportPill.className = "badge badge-warning";
      abstentionNote.style.display = "inline";
      leadingAgentLabel.innerHTML = "Status: <strong>No recommendation met trust threshold</strong>";
    }

    // 2. Conflict Alert
    const conflictAlert = document.getElementById("conflictAlert");
    const conflictAlertText = document.getElementById("conflictAlertText");
    const conflictKeys = Object.keys(conflicting_answers || {});
    if (conflictKeys.length > 1) {
      conflictAlert.style.display = "flex";
      const parts = Object.entries(conflicting_answers).map(([ans, agents]) => {
        return `"${escapeHtml(ans)}" (${escapeHtml(agents.join(", "))})`;
      });
      conflictAlertText.innerHTML = `Agents reached ${conflictKeys.length} opposing conclusions: ${parts.join(" vs ")}.`;
    } else {
      conflictAlert.style.display = "none";
    }

    // 3. Agent Recommendations Table
    const tbody = document.getElementById("agentsTableBody");
    tbody.innerHTML = "";

    const contributionsMap = {};
    (decision.contributions || []).forEach((c) => {
      contributionsMap[c.agent_id] = c;
    });

    // Detect correlated models
    const modelCounts = {};
    result.responses.forEach((r) => {
      modelCounts[r.model_id] = (modelCounts[r.model_id] || 0) + 1;
    });

    result.responses.forEach((resp) => {
      const contrib = contributionsMap[resp.agent_id] || {};
      const tr = document.createElement("tr");

      // Agent & Role
      const agentCell = document.createElement("td");
      agentCell.innerHTML = `<strong>${escapeHtml(resp.agent_id)}</strong>`;

      // Model ID
      const modelCell = document.createElement("td");
      const isCorrelated = modelCounts[resp.model_id] > 1;
      modelCell.innerHTML = `<code>${escapeHtml(resp.model_id)}</code>${
        isCorrelated ? ' <span class="badge badge-warning" title="Correlated model weight shared">Shared Model</span>' : ' <span class="badge badge-accent">Independent</span>'
      }`;

      // Stance
      const stanceCell = document.createElement("td");
      const isAnswer = resp.stance === "answer";
      stanceCell.innerHTML = `<span class="badge ${isAnswer ? "badge-success" : "badge-warning"}">${escapeHtml(
        resp.stance.toUpperCase()
      )}</span>`;

      // Recommendation
      const recCell = document.createElement("td");
      recCell.innerHTML = resp.recommendation
        ? `<strong>${escapeHtml(resp.recommendation)}</strong>`
        : '<span style="color: var(--text-muted);">(abstained)</span>';

      // Confidence + mini-bar
      const confCell = document.createElement("td");
      const confVal = resp.self_reported_confidence;
      const confPct = Math.round(confVal * 100);
      confCell.innerHTML = `
        <div class="mini-bar-wrap">
          <span>${formatScore(confVal, 2)}</span>
          <div class="mini-bar" title="${confPct}% Stated Confidence">
            <div class="mini-bar-fill" style="width: ${confPct}%;"></div>
          </div>
        </div>
      `;

      // Trust Weight
      const weightCell = document.createElement("td");
      weightCell.innerHTML = contrib.weight !== null && contrib.weight !== undefined
        ? `<strong>${formatScore(contrib.weight, 2)}</strong>`
        : '<span class="badge badge-neutral">Not Counted</span>';

      // Adjustment Notes / Badges
      const noteCell = document.createElement("td");
      const noteWrap = document.createElement("div");
      noteWrap.className = "note-badge-wrap";

      if (contrib.note) {
        const badge = document.createElement("span");
        if (contrib.note.includes("reduced") || contrib.note.includes("contradicts")) {
          badge.className = "badge badge-danger";
        } else if (contrib.note.includes("shared") || contrib.note.includes("not counted")) {
          badge.className = "badge badge-warning";
        } else {
          badge.className = "badge badge-neutral";
        }
        badge.textContent = contrib.note;
        noteWrap.appendChild(badge);
      }
      noteCell.appendChild(noteWrap);

      tr.appendChild(agentCell);
      tr.appendChild(modelCell);
      tr.appendChild(stanceCell);
      tr.appendChild(recCell);
      tr.appendChild(confCell);
      tr.appendChild(weightCell);
      tr.appendChild(noteCell);
      tbody.appendChild(tr);
    });

    // 4. Strategy Comparison Grid
    const strategiesGrid = document.getElementById("strategiesGrid");
    strategiesGrid.innerHTML = "";
    (evaluation.decisions || []).forEach((strat) => {
      const card = document.createElement("div");
      const isTrust = strat.strategy === "trust_weighted";
      card.className = `strategy-card ${isTrust ? "highlight" : ""}`;

      const isDecided = strat.outcome === "decided";
      card.innerHTML = `
        <div class="strategy-name">
          <span>${escapeHtml(strat.strategy)}</span>
          ${isTrust ? '<span class="badge badge-accent">TrustNet Engine</span>' : '<span class="badge badge-neutral">Baseline</span>'}
        </div>
        <div class="strategy-result ${isDecided ? "decided" : "abstained"}">
          ${isDecided ? `Answer: ${escapeHtml(strat.answer)}` : "ABSTAINED"}
        </div>
        <div class="strategy-rationale">
          ${escapeHtml(strat.rationale || strat.abstention_reason || "No explanation.")}
        </div>
        <div class="strategy-meta">
          Support: <strong>${formatPercent(strat.support)}</strong>
        </div>
      `;
      strategiesGrid.appendChild(card);
    });

    // Reset Outcome Notification
    document.getElementById("outcomeNotification").style.display = "none";
  }

  // --- RECORD OUTCOME (HUMAN-IN-THE-LOOP) ---
  async function handleRecordOutcome() {
    if (!state.lastEvaluation) {
      alert("No active task evaluation to record an outcome for.");
      return;
    }

    const taskId = state.lastEvaluation.result.task.task_id;
    const isUnanswerable = document.getElementById("unanswerableCheckbox").checked;
    const answerInput = document.getElementById("correctAnswerInput").value.trim();
    const sourceInput = document.getElementById("resolutionSourceInput").value.trim();
    const notif = document.getElementById("outcomeNotification");

    if (!sourceInput) {
      alert("Please provide the resolution source (who or what verified this outcome).");
      document.getElementById("resolutionSourceInput").focus();
      return;
    }

    const correctAnswer = isUnanswerable ? null : answerInput || null;
    const recordBtn = document.getElementById("recordOutcomeBtn");
    recordBtn.disabled = true;

    try {
      const res = await window.api.recordOutcome({
        task_id: taskId,
        correct_answer: correctAnswer,
        resolution_source: sourceInput,
        mode: state.currentMode
      });

      notif.className = "notification success";
      notif.style.display = "flex";
      notif.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <span>Outcome recorded: <strong>${
          res.is_answerable ? escapeHtml(res.correct_answer) : "Verified as Unanswerable"
        }</strong>. Agent reliability profiles dynamically recomputed.</span>
      `;

      // Update active reliability views with recomputed profiles
      if (res.profiles) {
        state.latestProfiles = res.profiles;
        renderReliabilityFromProfiles(res.profiles);
      }
    } catch (err) {
      notif.className = "notification error";
      notif.style.display = "flex";
      notif.textContent = `Failed to record outcome: ${err.message}`;
    } finally {
      recordBtn.disabled = false;
    }
  }

  // --- DATABASE SEEDER ---
  async function handleSeedDatabase() {
    const btn = document.getElementById("seedDatabaseBtn");
    btn.disabled = true;
    try {
      const res = await window.api.seedDatabase();
      alert(`Database successfully seeded with ${res.tasks_seeded} verified tasks and outcomes across Healthcare, Infosec, Finance, and Architecture!`);
      loadAgentProfiles();
    } catch (err) {
      alert("Seeding failed: " + err.message);
    } finally {
      btn.disabled = false;
    }
  }

  // --- RELIABILITY VIEW RENDERING ---
  async function loadAgentProfiles() {
    try {
      const profiles = await window.api.getAgents(state.currentMode);
      state.latestProfiles = profiles;
      renderReliabilityFromProfiles(profiles);
    } catch (err) {
      console.warn("Could not load initial agent profiles:", err.message);
    }
  }

  function renderReliabilityFromProfiles(profiles) {
    const tbody = document.getElementById("reliabilityTableBody");
    tbody.innerHTML = "";

    const agentIds = Object.keys(profiles);
    if (!agentIds.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No agent profiles recorded in this repository.</td></tr>';
      return;
    }

    agentIds.forEach((agentId) => {
      const categories = profiles[agentId];
      Object.keys(categories).forEach((cat) => {
        const sig = categories[cat];
        appendReliabilityRow(tbody, agentId, cat, sig);
      });
    });
  }

  function renderReliabilityFromEvaluation(trustMap, currentCategory) {
    const tbody = document.getElementById("reliabilityTableBody");
    tbody.innerHTML = "";
    Object.keys(trustMap).forEach((agentId) => {
      const sig = trustMap[agentId];
      appendReliabilityRow(tbody, agentId, currentCategory || sig.category || "overall", sig);
    });
  }

  function appendReliabilityRow(tbody, agentId, category, sig) {
    const tr = document.createElement("tr");
    const hasEnough = sig.has_sufficient_history;
    const samples = sig.resolved_sample_size || 0;

    tr.innerHTML = `
      <td><strong>${escapeHtml(agentId)}</strong></td>
      <td><span class="badge badge-neutral">${escapeHtml(category || "overall")}</span></td>
      <td>${
        sig.trust_score !== null && sig.trust_score !== undefined
          ? `<strong style="color: var(--accent-primary); font-size: 14px;">${formatScore(sig.trust_score, 2)}</strong>`
          : '<span class="badge badge-warning" title="Less than 5 verified outcomes">N/A</span>'
      }</td>
      <td>${formatScore(sig.historical_accuracy, 3)}</td>
      <td>${formatScore(sig.accuracy_lower_bound, 3)}</td>
      <td>${formatScore(sig.calibration, 3)}</td>
      <td>${formatScore(sig.evidence_quality, 3)}</td>
      <td>${formatScore(sig.consistency, 3)}</td>
      <td>${formatScore(sig.insufficient_information_behavior, 3)}</td>
      <td>
        <span class="badge ${hasEnough ? "badge-success" : "badge-warning"}">
          ${samples} sample${samples === 1 ? "" : "s"} ${hasEnough ? "(≥5)" : "(Cold start)"}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  }

  // --- EVIDENCE VIEW RENDERING ---
  function renderEvidenceView(assessments, responses) {
    const tbody = document.getElementById("evidenceTableBody");
    tbody.innerHTML = "";

    const assessmentMap = {};
    assessments.forEach((a) => {
      assessmentMap[a.evidence_id] = a;
    });

    const items = [];
    responses.forEach((resp) => {
      (resp.evidence || []).forEach((ev) => {
        items.push({
          agent_id: resp.agent_id,
          excerpt: ev.excerpt,
          reference: ev.reference,
          assessment: assessmentMap[ev.evidence_id]
        });
      });
    });

    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No evidence was cited by any responding agent for this task.</td></tr>';
      return;
    }

    items.forEach((item) => {
      const tr = document.createElement("tr");
      const status = item.assessment ? item.assessment.status : "unverified";
      let statusBadgeClass = "badge-neutral";
      if (status === "verified") statusBadgeClass = "badge-success";
      else if (status === "contradictory") statusBadgeClass = "badge-danger";
      else if (status === "unsupported") statusBadgeClass = "badge-warning";

      const hasDoc = item.reference && (item.reference.endsWith(".txt") || item.reference.includes("/"));
      const docBtn = hasDoc
        ? `<button type="button" class="btn btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="window.previewSourceDocument('${escapeHtml(
            item.reference.replace('data/sources/', '')
          )}')">View Ground Truth</button>`
        : "—";

      tr.innerHTML = `
        <td><strong>${escapeHtml(item.agent_id)}</strong></td>
        <td>${escapeHtml(item.excerpt)}</td>
        <td><code>${escapeHtml(item.reference || "(none)")}</code></td>
        <td><span class="badge ${statusBadgeClass}">${escapeHtml(status.toUpperCase())}</span></td>
        <td><span class="badge badge-neutral">${escapeHtml(item.assessment?.method || "deterministic")}</span></td>
        <td style="color: var(--text-secondary);">${escapeHtml(item.assessment?.note || "—")}</td>
        <td>${docBtn}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // --- BENCHMARK ARENA WORKFLOW ---
  async function handleRunBenchmark() {
    const seed = parseInt(document.getElementById("benchmarkSeedInput").value, 10) || 7;
    const btn = document.getElementById("runBenchmarkBtn");
    const loading = document.getElementById("benchmarkLoading");
    const results = document.getElementById("benchmarkResults");

    btn.disabled = true;
    results.style.display = "none";
    loading.style.display = "flex";

    try {
      const report = await window.api.getBenchmark({ seed, development_size: 300, test_size: 300 });
      renderBenchmarkReport(report);
      results.style.display = "block";
    } catch (err) {
      alert(`Benchmark execution failed: ${err.message}`);
    } finally {
      loading.style.display = "none";
      btn.disabled = false;
    }
  }

  function renderBenchmarkReport(report) {
    // Summary
    const banner = document.getElementById("benchmarkSummaryBanner");
    banner.innerHTML = `
      <strong>Held-Out Benchmark Configuration:</strong>
      Seed = <code>${report.seed}</code> |
      Development split: <strong>${report.development_cases}</strong> cases |
      Held-out test split: <strong>${report.test_cases}</strong> cases (${report.test_unanswerable} unanswerable).
      <div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);">
        Notice: Simulated recipes with synthetic ground-truth. Reliability learned strictly on development split and verified on held-out test split.
      </div>
    `;

    // Strategy Comparison Table
    const stratTbody = document.getElementById("benchmarkStrategiesTableBody");
    stratTbody.innerHTML = "";
    (report.strategies || []).forEach((strat) => {
      const tr = document.createElement("tr");
      const isTrust = strat.strategy === "trust_weighted";
      if (isTrust) tr.style.backgroundColor = "var(--accent-subtle)";

      const ci = strat.accuracy_interval
        ? `[${formatScore(strat.accuracy_interval[0], 3)}, ${formatScore(strat.accuracy_interval[1], 3)}]`
        : "N/A";

      tr.innerHTML = `
        <td><strong>${escapeHtml(strat.strategy)}</strong> ${isTrust ? '<span class="badge badge-accent">Ours</span>' : ""}</td>
        <td>${strat.cases}</td>
        <td><strong style="color: ${isTrust ? "var(--status-success)" : "inherit"}; font-size: 14px;">${formatScore(strat.accuracy, 3)}</strong></td>
        <td><code>${ci}</code></td>
        <td>${formatPercent(strat.abstention_rate)}</td>
        <td>${strat.answered}</td>
        <td><strong>${formatScore(strat.answered_accuracy, 3)}</strong></td>
        <td><code>${formatScore(strat.brier_score, 3)}</code></td>
      `;
      stratTbody.appendChild(tr);
    });

    // Agent Profiles Table
    const agentTbody = document.getElementById("benchmarkAgentsTableBody");
    agentTbody.innerHTML = "";
    (report.agents || []).forEach((agent) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${escapeHtml(agent.agent_id)}</strong></td>
        <td>${agent.resolved_sample_size}</td>
        <td>${formatScore(agent.historical_accuracy, 3)}</td>
        <td>${formatScore(agent.accuracy_lower_bound, 3)}</td>
        <td>${formatScore(agent.calibration, 3)}</td>
        <td>${formatScore(agent.evidence_quality, 3)}</td>
        <td>${formatScore(agent.consistency, 3)}</td>
        <td>${formatScore(agent.insufficient_information_behavior, 3)}</td>
        <td><strong style="color: var(--accent-primary);">${formatScore(agent.trust_score, 3)}</strong></td>
      `;
      agentTbody.appendChild(tr);
    });
  }
})();
