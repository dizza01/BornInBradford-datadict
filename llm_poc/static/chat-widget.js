/* ── BiB Research Assistant — Floating Chat Widget ────────────────────────── */
(function () {
  "use strict";

  // Don't double-inject
  if (document.getElementById("bib-chat-btn")) return;

  // ── Conversation history (for context threading) ──────────────────────────
  const convHistory = [];

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

  // ── Message helpers ───────────────────────────────────────────────────────
  const msgs = document.getElementById("bib-messages");
  let thinkingEl = null;

  function escHtml(t) {
    return String(t)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
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
          if (evt.done) {
            if (msgEl) msgEl.innerHTML = formatAnswer(fullText);
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
