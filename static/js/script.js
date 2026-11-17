/**
 * PromptShield AI — Frontend Logic
 * Handles prompt submission, loading states, and result rendering.
 */

// ── DOM References ───────────────────────────────────────────
const promptInput    = document.getElementById("prompt-input");
const analyzeBtn     = document.getElementById("analyze-btn");
const clearBtn       = document.getElementById("clear-btn");
const spinner        = document.getElementById("btn-spinner");
const btnText        = document.getElementById("btn-text");
const charCounter    = document.getElementById("char-counter");
const resultsSection = document.getElementById("results-section");
const initState      = document.getElementById("init-state");
const analysisResult = document.getElementById("analysis-result");
const typewriterEl   = document.getElementById("typewriter-text");

// ── Typewriter effect on init state ─────────────────────────
const typewriterMessages = [
  "Security analysis engine active...",
  "Awaiting prompt input...",
  "XAI model loaded and ready...",
  "Ready to detect prompt injections and vulnerabilities...",
];
let twIndex = 0;
let twCharIndex = 0;
let twTimeout;

function typewrite() {
  if (!typewriterEl) return;
  const msg = typewriterMessages[twIndex];
  if (twCharIndex <= msg.length) {
    typewriterEl.textContent = msg.slice(0, twCharIndex);
    twCharIndex++;
    twTimeout = setTimeout(typewrite, 50);
  } else {
    setTimeout(() => {
      twCharIndex = 0;
      twIndex = (twIndex + 1) % typewriterMessages.length;
      typewrite();
    }, 2000);
  }
}
typewrite();

// ── Character counter ────────────────────────────────────────
promptInput.addEventListener("input", () => {
  const len = promptInput.value.length;
  charCounter.textContent = `${len} chars`;
  charCounter.style.color = len > 2000 ? "var(--accent-amber)" : "var(--text-muted)";
});

// ── Clear button ─────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  promptInput.value = "";
  charCounter.textContent = "0 chars";
  hideResults();
  promptInput.focus();
});

// ── Analyze button & Keyboard shortcut ───────────────────────
analyzeBtn.addEventListener("click", handleAnalyze);

promptInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") handleAnalyze();
});

async function handleAnalyze() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    showError("Please enter a prompt to analyze.");
    promptInput.focus();
    return;
  }
  setLoading(true);
  hideResults();
  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    if (!response.ok || data.status === "error") {
      throw new Error(data.error || `Server error: ${response.status}`);
    }
    renderResult(data);
    saveScanToHistory(data);
    updateDashboard();
  } catch (err) {
    if (err.name === "TypeError") {
      showError("Cannot reach the server. Make sure the Flask app is running at http://127.0.0.1:5000.");
    } else {
      showError(err.message || "An unexpected error occurred during analysis.");
    }
  } finally {
    setLoading(false);
  }
}

// ── Loading state ─────────────────────────────────────────────
function setLoading(active) {
  analyzeBtn.disabled = active;
  spinner.style.display = active ? "block" : "none";
  btnText.textContent   = active ? "Analyzing..." : "Analyze Prompt";
}

// ── Show results ─────────────────────────────────────────────
function renderResult(data) {
  analysisResult.innerHTML = "";

  const category   = data.category    || "Unknown";
  const riskScore  = typeof data.risk_score === "number" ? data.risk_score : 0;
  const severity   = (data.severity   || "LOW").toUpperCase();
  const confidence = typeof data.confidence === "number" ? data.confidence.toFixed(2) : "0.00";
  const ruleMatch  = !!data.rule_match;
  const secInd     = data.security_indicator || "";
  const topInd     = Array.isArray(data.top_indicators) ? data.top_indicators : [];
  const methods    = Array.isArray(data.analysis_methods) ? data.analysis_methods : [];
  const breakdown  = data.risk_breakdown || {};
  const details    = data.analysis_details || "";
  const isSafe     = category === "Safe Prompt";

  // Map pill status
  let pillStatus = "safe";
  if (severity === "CRITICAL" || severity === "HIGH") pillStatus = "danger";
  else if (severity === "MEDIUM") pillStatus = "warning";

  // Color helpers
  const categoryColor = isSafe ? "green" : "cyan";
  const scoreColor    = riskScore <= 29 ? "green" : (riskScore <= 59 ? "amber" : "red");
  const severityColor = severity === "LOW" ? "green" : (severity === "MEDIUM" ? "amber" : "red");
  const meterClass    = riskScore <= 29 ? "low" : (riskScore <= 59 ? "medium" : (riskScore <= 79 ? "high" : "critical"));
  const scoreColorVar = riskScore <= 29 ? "var(--accent-green)" : (riskScore <= 59 ? "var(--accent-amber)" : "var(--accent-red)");

  const statusPill = createStatusPill(pillStatus, severity === "LOW" ? "Safe" : `${severity} Risk`);

  // Recommendation styling
  const recClass = isSafe ? "safe" : (severity === "MEDIUM" ? "warning" : "danger");
  const recIcon  = isSafe ? "✅" : (severity === "MEDIUM" ? "⚠️" : "🚫");

  // Indicator tags
  const tagsHtml = topInd.length > 0
    ? `<div class="indicator-tags">${topInd.map(w => `<span class="tag">${escapeHtml(w)}</span>`).join("")}</div><div style="color:var(--text-muted); font-size:0.78rem; margin-top:6px;">Contributing textual features, not definitive proof of intent.</div>`
    : `<span style="color:var(--text-muted); font-size:0.83rem;">No significant security indicator detected.</span>`;

  // Methods list
  const methodsHtml = methods.length > 0
    ? `<div class="methods-list">${methods.map(m => `<div class="method-item">${escapeHtml(m)}</div>`).join("")}</div>`
    : "";

  // Security indicator block
  const indicatorColor = isSafe ? "var(--accent-green)" : "var(--accent-amber)";
  const indicatorHtml  = secInd
    ? `<div class="flagged-section-content" style="color:${indicatorColor}; font-weight:600;">${escapeHtml(secInd)}</div>`
    : `<div class="flagged-section-content" style="color:var(--text-muted);">No significant security indicator detected.</div>`;

  // Flagged panel title
  const flaggedTitle = isSafe
    ? `<div class="flagged-panel-title">✅ Why is this considered safe?</div>`
    : `<div class="flagged-panel-title">🚨 Why was this flagged?</div>`;

  // Risk breakdown values
  const bdBase  = breakdown.base_category_risk          !== undefined ? breakdown.base_category_risk          : "—";
  const bdConf  = breakdown.model_confidence_adjustment !== undefined ? breakdown.model_confidence_adjustment : "—";
  const bdRule  = breakdown.security_rule_adjustment    !== undefined ? breakdown.security_rule_adjustment    : "—";
  const bdFinal = breakdown.final_score                 !== undefined ? breakdown.final_score                 : riskScore;

  const confSign = typeof bdConf === "number" ? (bdConf >= 0 ? `+${bdConf}` : `${bdConf}`) : bdConf;
  const ruleSign = typeof bdRule === "number" ? (bdRule >= 0 ? `+${bdRule}` : `${bdRule}`) : bdRule;

  // Rule match badge in engine status
  const ruleMatchBadge = ruleMatch
    ? `<span style="color:var(--accent-red); font-size:0.7rem; margin-left:6px;">RULE HIT</span>`
    : "";

  // ── Build HTML ───────────────────────────────────────────────
  analysisResult.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; flex-wrap:wrap; gap:10px;">
      <div class="section-title" style="margin-bottom:0; flex:1;">Security Analysis Report</div>
      ${statusPill}
    </div>

    <!-- Metric Cards -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-title">Threat Category</div>
        <div class="metric-value ${categoryColor}" id="metric-category" style="font-size:1.15rem; line-height:1.2;">${escapeHtml(category)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Risk Score</div>
        <div class="metric-value ${scoreColor}" id="metric-score">${riskScore}<span style="font-size:0.9rem; color:var(--text-muted);">/100</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Severity</div>
        <div class="metric-value ${severityColor}" id="metric-severity">${escapeHtml(severity)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value cyan" id="metric-confidence">${confidence}%</div>
      </div>
    </div>

    <!-- Risk Meter -->
    <div class="risk-meter-wrapper detail-block" style="padding:18px 20px;">
      <div class="risk-meter-header">
        <span class="risk-meter-label">Risk Score Meter</span>
        <span class="risk-meter-score" style="color:${scoreColorVar};">${riskScore} / 100</span>
      </div>
      <div class="risk-meter" role="progressbar" aria-valuenow="${riskScore}" aria-valuemin="0" aria-valuemax="100" aria-label="Risk score meter">
        <div class="risk-meter-fill ${meterClass}" id="risk-fill" style="width:0%;"></div>
      </div>
    </div>

    <!-- Engine Status -->
    <div class="engine-status-panel">
      <div class="engine-status-title">Analysis Engine Status</div>
      <div class="engine-status-grid">
        <div class="engine-row"><span class="engine-name">ML Classifier</span><span class="engine-check">&#10003; ACTIVE</span></div>
        <div class="engine-row"><span class="engine-name">Rule Engine${ruleMatchBadge}</span><span class="engine-check">&#10003; ACTIVE</span></div>
        <div class="engine-row"><span class="engine-name">Risk Engine</span><span class="engine-check">&#10003; ACTIVE</span></div>
        <div class="engine-row"><span class="engine-name">Explainability</span><span class="engine-check">&#10003; ACTIVE</span></div>
      </div>
    </div>

    <!-- Submitted Prompt -->
    <div class="detail-block">
      <div class="detail-block-label">Submitted Prompt</div>
      <div class="detail-block-content" id="detail-prompt" style="font-family:var(--font-mono); font-size:0.88rem; white-space:pre-wrap; word-break:break-word;">${escapeHtml(data.prompt || "")}</div>
    </div>

    <!-- Why was this flagged? -->
    <div class="flagged-panel ${isSafe ? "safe" : ""}">
      ${flaggedTitle}
      <div class="flagged-section">
        <div class="flagged-section-label">Security Indicator</div>
        ${indicatorHtml}
      </div>
      <div class="flagged-section">
        <div class="flagged-section-label">Contributing Textual Features</div>
        ${tagsHtml}
      </div>
      <div class="flagged-section">
        <div class="flagged-section-label">Analysis Pipeline</div>
        ${methodsHtml}
      </div>
      <div class="flagged-section">
        <div class="flagged-section-label">Explanation</div>
        <div class="flagged-section-content">${escapeHtml(details || data.explanation || "")}</div>
      </div>
    </div>

    <!-- Risk Breakdown -->
    <div class="breakdown-panel">
      <div class="breakdown-title">Risk Score Breakdown</div>
      <div class="breakdown-grid">
        <div class="breakdown-item">
          <div class="breakdown-item-label">Base Category Risk</div>
          <div class="breakdown-item-value">${bdBase}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-item-label">Model Confidence Adjustment</div>
          <div class="breakdown-item-value">${confSign}</div>
        </div>
        <div class="breakdown-item">
          <div class="breakdown-item-label">Rule Match Adjustment</div>
          <div class="breakdown-item-value">${ruleSign}</div>
        </div>
        <div class="breakdown-item final">
          <div class="breakdown-item-label">Final Risk Score</div>
          <div class="breakdown-item-value">${bdFinal}</div>
        </div>
      </div>
    </div>

    <!-- Recommendation -->
    <div class="recommendation-panel ${recClass}">
      <div class="recommendation-icon">${recIcon}</div>
      <div class="recommendation-body">
        <div class="recommendation-label">Recommended Action</div>
        <div class="recommendation-text" id="detail-recommendation">${escapeHtml(data.recommendation || "Review prompt against application safety policies.")}</div>
      </div>
    </div>

    <!-- Engine Message -->
    <div class="detail-block">
      <div class="detail-block-label">Engine Message</div>
      <div class="detail-block-content" id="detail-message">${escapeHtml(data.message || "Analysis complete.")}</div>
    </div>
  `;

  // Animate risk meter fill
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const fill = document.getElementById("risk-fill");
      if (fill) fill.style.width = `${riskScore}%`;
    });
  });

  initState.style.display = "none";
  resultsSection.style.display = "block";
  resultsSection.classList.add("visible");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Show error ────────────────────────────────────────────────
function showError(message) {
  analysisResult.innerHTML = `
    <div class="error-card">
      <span style="font-size:1.4rem;">&#9888;&#65039;</span>
      <div><strong>Validation or Analysis Error</strong><br><span>${escapeHtml(message)}</span></div>
    </div>
  `;
  initState.style.display = "none";
  resultsSection.style.display = "block";
  resultsSection.classList.add("visible");
}

// ── Hide results ──────────────────────────────────────────────
function hideResults() {
  resultsSection.style.display = "none";
  resultsSection.classList.remove("visible");
  initState.style.display = "block";
}

// ── Helpers ───────────────────────────────────────────────────
function createStatusPill(status, customLabel) {
  const labels = { pending: "Pending", safe: customLabel || "Safe", danger: customLabel || "Threat Detected", warning: customLabel || "Warning" };
  const icons  = { pending: "&#9203;", safe: "&#9989;", danger: "&#128680;", warning: "&#9888;&#65039;" };
  const cls    = labels[status] ? status : "pending";
  return `<span class="status-pill ${cls}">${icons[cls]} ${escapeHtml(labels[cls])}</span>`;
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Local Storage & Dashboard ─────────────────────────────────
const HISTORY_KEY = "promptshield_history";
const MAX_HISTORY = 20;

function saveScanToHistory(data) {
  let history = [];
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) history = JSON.parse(stored);
  } catch (e) {
    console.error("Could not read from local storage", e);
  }

  const record = {
    prompt: data.prompt,
    category: data.category || "Unknown",
    risk_score: typeof data.risk_score === "number" ? data.risk_score : 0,
    severity: (data.severity || "LOW").toUpperCase(),
    confidence: typeof data.confidence === "number" ? data.confidence : 0,
    timestamp: new Date().toISOString()
  };

  history.unshift(record);
  if (history.length > MAX_HISTORY) {
    history = history.slice(0, MAX_HISTORY);
  }

  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch (e) {
    console.error("Could not save to local storage", e);
  }
}

function updateDashboard() {
  let history = [];
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) history = JSON.parse(stored);
  } catch (e) {
    console.error("Could not read from local storage", e);
  }

  const totalScans = history.length;
  let safePrompts = 0;
  let threatsDetected = 0;
  let totalRisk = 0;

  const distribution = {
    "Safe Prompt": 0,
    "Prompt Injection": 0,
    "Jailbreak Attempts": 0,
    "Instruction Manipulation": 0,
    "Unsafe Instructions": 0,
    "Potential Data Leakage": 0
  };

  history.forEach(item => {
    if (item.category === "Safe Prompt") {
      safePrompts++;
    } else {
      threatsDetected++;
    }
    totalRisk += item.risk_score;
    
    if (distribution[item.category] !== undefined) {
      distribution[item.category]++;
    }
  });

  const avgRisk = totalScans > 0 ? Math.round(totalRisk / totalScans) : 0;

  // Update Stats
  const elTotalScans = document.getElementById("dash-total-scans");
  const elSafePrompts = document.getElementById("dash-safe-prompts");
  const elThreats = document.getElementById("dash-threats");
  const elAvgRisk = document.getElementById("dash-avg-risk");

  if (elTotalScans) elTotalScans.textContent = totalScans;
  if (elSafePrompts) elSafePrompts.textContent = safePrompts;
  if (elThreats) elThreats.textContent = threatsDetected;
  if (elAvgRisk) elAvgRisk.textContent = avgRisk;

  // Update Threat Distribution
  const distContainer = document.getElementById("threat-distribution");
  if (distContainer) {
    if (totalScans === 0) {
      distContainer.innerHTML = `<div class="dist-empty">No data available</div>`;
    } else {
      let distHtml = "";
      for (const [cat, count] of Object.entries(distribution)) {
        if (count > 0 || cat === "Safe Prompt" || cat === "Prompt Injection") { // Show a few key ones always or if >0
          const pct = Math.round((count / totalScans) * 100);
          const color = cat === "Safe Prompt" ? "var(--accent-green)" : "var(--accent-red)";
          distHtml += `
            <div class="dist-bar-wrapper">
              <div class="dist-label">${escapeHtml(cat)}</div>
              <div class="dist-track">
                <div class="dist-fill" style="width:${pct}%; background:${color};"></div>
              </div>
              <div class="dist-count">${count}</div>
            </div>
          `;
        }
      }
      distContainer.innerHTML = distHtml;
    }
  }

  // Render History
  renderHistory(history);
}

function renderHistory(history) {
  const container = document.getElementById("history-list");
  if (!container) return;

  if (history.length === 0) {
    container.innerHTML = `<div class="history-empty">No recent scans found. Analyze a prompt to see it here.</div>`;
    return;
  }

  let html = "";
  history.forEach(item => {
    const isSafe = item.category === "Safe Prompt";
    const scoreColor = item.risk_score <= 29 ? "var(--accent-green)" : (item.risk_score <= 59 ? "var(--accent-amber)" : "var(--accent-red)");
    const dateStr = new Date(item.timestamp).toLocaleString();
    
    html += `
      <div class="history-item">
        <div class="history-prompt">${escapeHtml(item.prompt)}</div>
        <div class="history-meta">
          <span style="font-size:0.75rem; color:${isSafe ? "var(--accent-green)" : "var(--accent-amber)"};">${escapeHtml(item.category)}</span>
          <span style="font-size:0.75rem; color:${scoreColor}; font-weight:600;">${item.risk_score}/100</span>
          <span class="history-time">${dateStr}</span>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

const clearHistoryBtn = document.getElementById("clear-history-btn");
if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    updateDashboard();
  });
}

// Initialize
updateDashboard();
