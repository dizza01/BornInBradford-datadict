/* ── BiB Research Assistant — Floating Chat Widget ────────────────────────── */
(function () {
  "use strict";

  // Don't double-inject
  if (document.getElementById("bib-chat-btn")) return;

  // ── Conversation history (for context threading) ──────────────────────────
  const convHistory = [];
  const selectedVariables = new Map();
  const STORAGE_KEY = "bibSelectedVariables";

  // ── Build DOM ─────────────────────────────────────────────────────────────
  const panel = document.createElement("div");
  panel.id = "bib-chat-panel";
  panel.className = "hidden";
  panel.innerHTML = `
    <div id="bib-panel-header">
      <span class="icon">🔬</span>
      <span class="title">BiB Research Assistant</span>
      <a href="/assistant" title="Open full-screen assistant">Full screen ↗</a>
      <button id="bib-close-btn" title="Close">✕</button>
    </div>
    <div id="bib-messages">
      <div id="bib-welcome">
        <div class="bib-w-icon">🧬</div>
        <div>Ask about variables, tables, published papers, or analysis approaches using the Born in Bradford dataset.</div>
      </div>
    </div>
    <div id="bib-variable-basket">
      <div class="bib-basket-summary">
        <strong>Selected variables</strong>
        <span id="bib-basket-count">0</span>
      </div>
      <div id="bib-basket-list"></div>
      <div class="bib-basket-actions">
        <button id="bib-export-vars" type="button" disabled>Export CSV</button>
        <button id="bib-clear-vars" type="button" disabled>Clear</button>
      </div>
    </div>
    <div id="bib-input-bar">
      <textarea id="bib-input" rows="1"
        placeholder="Ask a question…"
        onkeydown="(function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();window._bibSend();}})(event)"
        oninput="(function(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,80)+'px';})(this)"
      ></textarea>
      <button id="bib-send" onclick="window._bibSend()" title="Send">&#9658;</button>
    </div>`;

  const btn = document.createElement("button");
  btn.id = "bib-chat-btn";
  btn.title = "BiB Research Assistant";
  btn.innerHTML = "🔬";

  document.body.appendChild(panel);
  document.body.appendChild(btn);

  // ── Toggle open/close ─────────────────────────────────────────────────────
  function openPanel() {
    panel.classList.remove("hidden");
    panel.classList.add("visible");
    btn.innerHTML = "✕";
    btn.title = "Close";
    document.getElementById("bib-input").focus();
  }
  function closePanel() {
    panel.classList.remove("visible");
    panel.classList.add("hidden");
    btn.innerHTML = "🔬";
    btn.title = "BiB Research Assistant";
  }

  btn.addEventListener("click", function () {
    panel.classList.contains("visible") ? closePanel() : openPanel();
  });
  document.getElementById("bib-close-btn").addEventListener("click", closePanel);
  document.getElementById("bib-export-vars").addEventListener("click", exportSelectedVariables);
  document.getElementById("bib-clear-vars").addEventListener("click", () => {
    selectedVariables.clear();
    saveSelectedVariables();
    renderBasket();
  });
  loadSelectedVariables();
  renderBasket();

  // ── Message helpers ───────────────────────────────────────────────────────
  const msgs = document.getElementById("bib-messages");
  let thinkingEl = null;

  function escHtml(t) {
    return String(t)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function csvEscape(value) {
    const text = String(value || "");
    if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function variableKey(v) {
    return v.variable_id || [v.table, v.variable].filter(Boolean).join(".");
  }

  function loadSelectedVariables() {
    try {
      const rows = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "[]");
      rows.forEach(v => {
        const key = variableKey(v);
        if (key) selectedVariables.set(key, v);
      });
    } catch {
      selectedVariables.clear();
    }
  }

  function saveSelectedVariables() {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(selectedVariables.values())));
  }

  function renderBasket() {
    const countEl = document.getElementById("bib-basket-count");
    const listEl = document.getElementById("bib-basket-list");
    const exportBtn = document.getElementById("bib-export-vars");
    const clearBtn = document.getElementById("bib-clear-vars");
    const rows = Array.from(selectedVariables.values());

    countEl.textContent = String(rows.length);
    exportBtn.disabled = rows.length === 0;
    clearBtn.disabled = rows.length === 0;

    if (!rows.length) {
      listEl.innerHTML = '<div class="bib-basket-empty">Add variables from chat results to export them.</div>';
      return;
    }

    listEl.innerHTML = rows.map(v => `
      <span class="bib-var-chip" title="${escHtml(v.variable_id || '')}">
        <code>${escHtml(v.variable || v.variable_id)}</code>
        <button type="button" data-remove-var="${escHtml(variableKey(v))}" title="Remove">x</button>
      </span>
    `).join("");

    listEl.querySelectorAll("[data-remove-var]").forEach(btn => {
      btn.addEventListener("click", () => {
        selectedVariables.delete(btn.getAttribute("data-remove-var"));
        saveSelectedVariables();
        renderBasket();
      });
    });
  }

  function addVariable(v) {
    const key = variableKey(v);
    if (!key) return;
    selectedVariables.set(key, v);
    saveSelectedVariables();
    renderBasket();
  }

  function exportSelectedVariables() {
    const rows = Array.from(selectedVariables.values());
    if (!rows.length) return;

    const headers = [
      "variable_id", "variable", "table", "label", "description",
      "type", "non_missing", "topic", "theme", "study_context"
    ];
    const csv = [
      headers.join(","),
      ...rows.map(row => headers.map(h => csvEscape(row[h])).join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "bib-selected-variables.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function renderVariablePicker(container, variables) {
    if (!container || !variables || !variables.length) return;
    const unique = [];
    const seen = new Set();
    variables.forEach(v => {
      const key = variableKey(v);
      if (key && !seen.has(key)) {
        seen.add(key);
        unique.push(v);
      }
    });
    if (!unique.length) return;

    const picker = document.createElement("div");
    picker.className = "bib-detected-vars";
    picker.innerHTML = `
      <div class="bib-detected-head">
        <span>${unique.length} variable${unique.length === 1 ? "" : "s"} found</span>
        <button type="button" class="bib-add-all-vars">Add all</button>
      </div>
      <div class="bib-detected-list">
        ${unique.map(v => `
          <button type="button" class="bib-detected-var" data-var-key="${escHtml(variableKey(v))}">
            <code>${escHtml(v.variable || v.variable_id)}</code>
            <span>${escHtml(v.label || v.table || "")}</span>
          </button>
        `).join("")}
      </div>
    `;
    container.appendChild(picker);

    picker.querySelector(".bib-add-all-vars").addEventListener("click", () => {
      unique.forEach(addVariable);
    });
    picker.querySelectorAll("[data-var-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        const row = unique.find(v => variableKey(v) === btn.getAttribute("data-var-key"));
        if (row) addVariable(row);
      });
    });
  }

  function renderVariableStudySummary(container, result) {
    const summary = result.study_summary || [];
    if (!container || !summary.length) return;

    const terms = (result.terms || []).slice(0, 8);
    const rows = result.rows || [];
    const rowByKey = new Map(rows.map(row => [variableKey(row), row]));
    const panel = document.createElement("div");
    panel.className = "bib-study-summary";
    panel.innerHTML = `
      <div class="bib-study-summary-head">
        <div>
          <strong>Variables by study/cohort</strong>
          <span>${result.total} matching variable${result.total === 1 ? "" : "s"} across ${summary.length} study context/cohort label${summary.length === 1 ? "" : "s"}</span>
        </div>
        ${terms.length ? `
          <div class="bib-study-term-pills">
            ${terms.map(term => `<span>${escHtml(term)}</span>`).join("")}
          </div>
        ` : ""}
      </div>
      <div class="bib-study-summary-list">
        ${summary.map((item, idx) => {
          const studyRows = rows.filter(row => row.study_context === item.study_context);
          const examples = studyRows.length
            ? studyRows
            : (item.examples || []).map(example => ({
                variable_id: example.variable_id,
                variable: example.variable_id,
                label: example.label,
              }));
          const nVars = Number(item.n_variables || 0);
          const nTables = Number(item.n_tables || 0);
          return `
            <div class="bib-study-card">
              <div class="bib-study-card-top">
                <div>
                  <strong>${escHtml(item.study_context || "Study not inferred")}</strong>
                  <span>${nVars} variable${nVars === 1 ? "" : "s"} · ${nTables} table${nTables === 1 ? "" : "s"}</span>
                </div>
                <button type="button" data-study-index="${idx}">Add cohort</button>
              </div>
              ${examples.length ? `
                <div class="bib-study-vars" aria-label="Variables in ${escHtml(item.study_context || "study")}">
                  ${examples.map(row => `
                    <button type="button" data-study-var-key="${escHtml(variableKey(row))}">
                      <code>${escHtml(row.variable_id || row.variable || "")}</code>
                      <span>${escHtml(row.label || row.table || "")}</span>
                    </button>
                  `).join("")}
                </div>
                ${studyRows.length && studyRows.length < nVars ? `<div class="bib-study-card-note">Showing ${studyRows.length} of ${nVars} variables for this cohort.</div>` : ""}
              ` : ""}
            </div>
          `;
        }).join("")}
      </div>
      <div class="bib-study-summary-note">
        The full matching variable set is shown below for review and CSV export.
      </div>
    `;
    container.appendChild(panel);

    panel.querySelectorAll("[data-study-index]").forEach(btn => {
      btn.addEventListener("click", () => {
        const item = summary[Number(btn.getAttribute("data-study-index"))];
        const study = item && item.study_context;
        rows
        .filter(row => row.study_context === study)
        .forEach(addVariable);
      });
    });
    panel.querySelectorAll("[data-study-var-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        const row = rowByKey.get(btn.getAttribute("data-study-var-key"));
        if (row) addVariable(row);
      });
    });
  }

  function renderVariableResults(container, result) {
    if (!container || !result || !result.rows || !result.rows.length) return;
    const rows = result.rows;
    const terms = (result.terms || []).slice(0, 10).join(", ");
    const filters = (result.study_filters || []).join(", ");

    const panel = document.createElement("div");
    panel.className = "bib-variable-results";
    panel.innerHTML = `
      <div class="bib-variable-results-head">
        <div>
          <strong>${result.total} variable${result.total === 1 ? "" : "s"} found</strong>
          <div class="bib-variable-results-meta">
            ${terms ? `Matched terms: ${escHtml(terms)}` : "Matched by registry filters"}
            ${filters ? ` · Study: ${escHtml(filters)}` : ""}
            ${result.truncated ? ` · Showing ${result.returned}` : ""}
          </div>
        </div>
        <button type="button" class="bib-add-all-results">Add all</button>
      </div>
      <div class="bib-variable-results-list">
        ${rows.map(v => `
          <button type="button" class="bib-variable-result" data-var-key="${escHtml(variableKey(v))}">
            <code>${escHtml(v.variable_id || v.variable)}</code>
            <span>${escHtml(v.label || v.table || "")}</span>
          </button>
        `).join("")}
      </div>
    `;
    container.appendChild(panel);

    panel.querySelector(".bib-add-all-results").addEventListener("click", () => {
      rows.forEach(addVariable);
    });
    panel.querySelectorAll("[data-var-key]").forEach(btn => {
      btn.addEventListener("click", () => {
        const row = rows.find(v => variableKey(v) === btn.getAttribute("data-var-key"));
        if (row) addVariable(row);
      });
    });
  }

  function renderMdTable(lines) {
    const rows = lines.filter(l => !l.trim().match(/^\|[-: |]+\|$/));
    if (!rows.length) return '';
    let html = '<table class="bib-md-table">';
    const headers = rows[0].trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
    html += '<thead><tr>' + headers.map(h => `<th>${escHtml(h)}</th>`).join('') + '</tr></thead>';
    if (rows.length > 1) {
      html += '<tbody>';
      for (let i = 1; i < rows.length; i++) {
        const cells = rows[i].trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
        html += '<tr>' + cells.map(c => `<td>${escHtml(c)}</td>`).join('') + '</tr>';
      }
      html += '</tbody>';
    }
    return html + '</table>';
  }
  function formatInline(l) {
    return escHtml(l)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
  }
  function formatLine(l) { return formatInline(l); }
  function formatAnswer(text) {
    const lines = text.split('\n');
    const segments = [];
    let textBuf = [];
    let i = 0;
    const flush = () => { if (textBuf.length) { segments.push({t:'text',l:textBuf.slice()}); textBuf=[]; } };
    while (i < lines.length) {
      const ln = lines[i];
      const hm = ln.match(/^(#{1,3}) (.+)/);
      if (hm) {
        flush();
        const tag = ['h3','h4','h5'][hm[1].length - 1];
        segments.push({t:'heading', tag, text: formatInline(hm[2])});
        i++;
      } else if (ln.trim().startsWith('|') && i+1 < lines.length && lines[i+1].trim().match(/^\|[-: |]+\|$/)) {
        flush();
        const tbl = [];
        while (i < lines.length && lines[i].trim().startsWith('|')) { tbl.push(lines[i++]); }
        segments.push({t:'table',l:tbl});
      } else { textBuf.push(ln); i++; }
    }
    flush();
    return segments.map(s => {
      if (s.t === 'table') return renderMdTable(s.l);
      if (s.t === 'heading') return `<${s.tag} class="bib-md-h">${s.text}</${s.tag}>`;
      return s.l.map(formatLine).join('<br>');
    }).join('');
  }

  function appendMsg(cls, html) {
    const welcome = document.getElementById("bib-welcome");
    if (welcome) welcome.remove();
    const div = document.createElement("div");
    div.className = "bib-msg " + cls;
    div.innerHTML = html;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function showThinking() {
    thinkingEl = appendMsg(
      "bib-thinking",
      'Searching… <span class="bib-dot-bounce"><span></span><span></span><span></span></span>'
    );
  }

  function removeThinking() {
    if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
  }

  // ── Send message ──────────────────────────────────────────────────────────
  window._bibSend = async function () {
    const input   = document.getElementById("bib-input");
    const sendBtn = document.getElementById("bib-send");
    const q = input.value.trim();
    if (!q) return;

    appendMsg("bib-user", escHtml(q));
    convHistory.push({ role: "user", content: q });
    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;
    showThinking();

    let msgEl    = null;
    let fullText = "";
    let detectedVariables = [];
    let variableResults = null;

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          history: convHistory.slice(0, -1),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Server error " + res.status }));
        removeThinking();
        appendMsg("bib-error", "⚠ " + escHtml(err.error || "Unknown error"));
        return;
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();  // keep incomplete trailing line
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }

          if (evt.token) {
            if (!msgEl) { removeThinking(); msgEl = appendMsg("bib-bot", ""); }
            fullText += evt.token;
            msgEl.textContent = fullText;  // plain text while streaming
            msgs.scrollTop = msgs.scrollHeight;
          }
          if (evt.replace) {
            fullText = evt.replace;
            if (msgEl) msgEl.innerHTML = formatAnswer(fullText);
            msgs.scrollTop = msgs.scrollHeight;
          }
          if (evt.error) {
            removeThinking();
            appendMsg("bib-error", "⚠ " + escHtml(evt.error));
          }
          if (evt.variables) {
            detectedVariables = evt.variables;
          }
          if (evt.variable_results) {
            variableResults = evt.variable_results;
          }
          if (evt.done) {
            if (!msgEl) { removeThinking(); msgEl = appendMsg("bib-bot", ""); }
            if (variableResults && variableResults.rows && variableResults.rows.length) {
              if (variableResults.summary_mode === "study_context" && variableResults.study_summary) {
                msgEl.innerHTML = "";
                renderVariableStudySummary(msgEl, variableResults);
              } else {
                msgEl.innerHTML = formatAnswer(fullText);
              }
              renderVariableResults(msgEl, variableResults);
            } else {
              msgEl.innerHTML = formatAnswer(fullText);
              renderVariablePicker(msgEl, detectedVariables);
            }
            convHistory.push({ role: "assistant", content: fullText });
            msgs.scrollTop = msgs.scrollHeight;
          }
        }
      }
    } catch (err) {
      removeThinking();
      appendMsg("bib-error", "⚠ Network error: " + escHtml(err.message));
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  };
})();
