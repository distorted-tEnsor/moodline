/**
 * Moodline front-end.
 *
 * Changes from the first version:
 *  - reads `probabilities` (falls back to the old misspelled key)
 *  - one in-flight request at a time, older ones aborted
 *  - health polling backs off instead of hammering every 3s, and pauses when
 *    the tab is hidden
 *  - results are built with textContent, never innerHTML
 *  - surfaces the model's own uncertainty instead of presenting every guess
 *    with the same confidence
 */
(() => {
  "use strict";

  const FALLBACK_EMOJI = {
    sadness: "😢", joy: "😄", love: "❤️",
    anger: "😠", fear: "😨", surprise: "😲",
  };

  const el = (id) => document.getElementById(id);
  const ui = {
    statusDot: el("statusDot"),
    serverStatusText: el("serverStatusText"),
    textInput: el("textInput"),
    charCount: el("charCount"),
    analyzeBtn: el("analyzeBtn"),
    errorMsg: el("errorMsg"),
    orb: el("orb"),
    orbEmoji: el("orbEmoji"),
    resultSection: el("resultSection"),
    emotionWord: el("emotionWord"),
    emotionEmoji: el("emotionEmoji"),
    confidenceText: el("confidenceText"),
    caveat: el("caveat"),
    echoedText: el("echoedText"),
    barsContainer: el("barsContainer"),
    facts: el("facts"),
    examples: el("examples"),
    historySection: el("historySection"),
    historyList: el("historyList"),
  };

  const state = {
    ready: false,
    emoji: { ...FALLBACK_EMOJI },
    inFlight: null,
    history: [],
    healthDelay: 1000,
    healthTimer: null,
  };

  /* ----------------------------------------------------------------
     Server state
  ---------------------------------------------------------------- */

  async function checkHealth() {
    clearTimeout(state.healthTimer);
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      const data = await response.json();

      state.ready = Boolean(data.model_loaded);
      if (state.ready) {
        setStatus("live", "model ready — say something");
        state.healthDelay = 1000;
        loadLabels();
      } else {
        setStatus("warming", "waking the model up…");
        scheduleHealth();
      }
    } catch {
      setStatus("down", "can't reach the server — retrying");
      scheduleHealth();
    }
    syncButtonState();
  }

  function scheduleHealth() {
    // 1s, 2s, 4s … capped at 30s, so a sleeping free-tier dyno is not hammered.
    state.healthDelay = Math.min(state.healthDelay * 2, 30000);
    state.healthTimer = setTimeout(checkHealth, state.healthDelay);
  }

  async function loadLabels() {
    try {
      const response = await fetch("/labels");
      if (!response.ok) return;
      for (const label of await response.json()) {
        if (label.emoji) state.emoji[label.name] = label.emoji;
      }
    } catch { /* cosmetic only */ }
  }

  function setStatus(kind, text) {
    ui.statusDot.className = "brand-mark " + kind;
    ui.serverStatusText.textContent = text;
  }

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.ready) {
      state.healthDelay = 1000;
      checkHealth();
    } else if (document.visibilityState === "hidden") {
      clearTimeout(state.healthTimer);
    }
  });

  /* ----------------------------------------------------------------
     Input
  ---------------------------------------------------------------- */

  ui.textInput.addEventListener("input", () => {
    ui.charCount.textContent = ui.textInput.value.length;
    syncButtonState();
  });

  ui.textInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runAnalysis();
    }
  });

  ui.examples.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    ui.textInput.value = chip.dataset.text;
    ui.charCount.textContent = ui.textInput.value.length;
    syncButtonState();
    ui.textInput.focus();
    runAnalysis();
  });

  ui.analyzeBtn.addEventListener("click", runAnalysis);

  function syncButtonState() {
    const hasText = ui.textInput.value.trim().length > 0;
    ui.analyzeBtn.disabled = !hasText || !state.ready || state.inFlight !== null;
  }

  /* ----------------------------------------------------------------
     Analysis
  ---------------------------------------------------------------- */

  async function runAnalysis() {
    const text = ui.textInput.value.trim();
    if (!text || !state.ready) return;

    if (state.inFlight) state.inFlight.abort();
    const controller = new AbortController();
    state.inFlight = controller;

    hideError();
    enterThinking();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      if (!response.ok) throw new Error(await describeFailure(response));

      const data = await response.json();
      state.inFlight = null;
      renderResult(data, text);
    } catch (error) {
      if (error.name === "AbortError") return;
      state.inFlight = null;
      exitThinking(false);
      showError(error.message || "Something went wrong. Try again.");
    }
  }

  async function describeFailure(response) {
    if (response.status === 429) return "Too many requests in a row — give it a moment.";
    if (response.status === 503) return "The model is still waking up. Try again in a few seconds.";
    const body = await response.json().catch(() => ({}));
    if (typeof body.detail === "string") return body.detail;
    return `Request failed (${response.status}).`;
  }

  function enterThinking() {
    ui.analyzeBtn.classList.add("loading");
    ui.analyzeBtn.querySelector(".btn-label").textContent = "Reading…";
    ui.analyzeBtn.disabled = true;
    ui.orb.classList.remove("settled");
    ui.orb.classList.add("thinking");
    ui.orbEmoji.style.opacity = "0";
  }

  function exitThinking(success) {
    ui.analyzeBtn.classList.remove("loading");
    ui.analyzeBtn.querySelector(".btn-label").textContent = "Read the mood";
    ui.orb.classList.remove("thinking");
    if (!success) {
      ui.orbEmoji.textContent = "✎";
      ui.orbEmoji.style.opacity = "1";
    }
    syncButtonState();
  }

  function renderResult(data, originalText) {
    const emotion = data.predicted_emotion;
    const emoji = state.emoji[emotion] || "🙂";
    const probabilities = data.probabilities || data.all_probabilites || {};

    document.body.setAttribute("data-emotion", emotion);

    ui.orb.classList.add("settled");
    ui.orbEmoji.textContent = emoji;
    ui.orbEmoji.style.opacity = "1";
    exitThinking(true);

    ui.emotionWord.textContent = capitalize(emotion);
    ui.emotionEmoji.textContent = emoji;
    ui.confidenceText.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
    ui.echoedText.textContent = `“${originalText}”`;

    renderCaveat(data, probabilities);
    renderBars(probabilities);
    renderFacts(data);
    pushHistory(emotion, emoji, originalText, data.confidence);

    ui.resultSection.hidden = false;
    ui.resultSection.classList.remove("entering");
    void ui.resultSection.offsetWidth; // restart the entrance animation
    ui.resultSection.classList.add("entering");
    ui.resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderCaveat(data, probabilities) {
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const [top, second] = sorted;
    let message = "";

    if (data.vocab_coverage !== undefined && data.vocab_coverage < 0.5) {
      message = "Most of these words are outside the model's 10k-word vocabulary, so treat this as a guess.";
    } else if (data.confidence < 0.4) {
      message = "The model is genuinely unsure — nothing here scored strongly.";
    } else if (second && top[1] - second[1] < 0.15) {
      message = `It is nearly split between ${top[0]} and ${second[0]}.`;
    }

    ui.caveat.textContent = message;
    ui.caveat.hidden = message === "";
  }

  function renderBars(probabilities) {
    const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    ui.barsContainer.replaceChildren();

    entries.forEach(([label, value], index) => {
      const percent = value * 100;

      const row = document.createElement("div");
      row.className = `bar-row bar-${label}`;

      const name = document.createElement("span");
      name.className = "bar-label";
      name.textContent = `${state.emoji[label] || ""} ${label}`;

      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = "bar-fill";
      track.appendChild(fill);

      const pct = document.createElement("span");
      pct.className = "bar-pct";
      pct.textContent = `${percent.toFixed(1)}%`;

      row.append(name, track, pct);
      ui.barsContainer.appendChild(row);

      requestAnimationFrame(() => {
        setTimeout(() => { fill.style.width = `${percent}%`; }, index * 60);
      });
    });
  }

  function renderFacts(data) {
    const facts = [
      ["tokens read", data.tokens],
      ["in vocabulary", data.vocab_coverage === undefined ? "—" : `${Math.round(data.vocab_coverage * 100)}%`],
      ["response", `${data.took_ms?.toFixed?.(1) ?? "—"} ms${data.cached ? " (cached)" : ""}`],
      ["model", data.model_version || "—"],
    ];

    ui.facts.replaceChildren();
    for (const [term, value] of facts) {
      if (value === undefined || value === null) continue;
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      ui.facts.append(dt, dd);
    }
  }

  function pushHistory(emotion, emoji, text, confidence) {
    state.history.unshift({ emotion, emoji, text, confidence });
    state.history = state.history.slice(0, 6);

    ui.historyList.replaceChildren();
    for (const item of state.history) {
      const li = document.createElement("li");
      li.className = `history-item bar-${item.emotion}`;

      const face = document.createElement("span");
      face.className = "history-emoji";
      face.textContent = item.emoji;

      const quote = document.createElement("span");
      quote.className = "history-text";
      quote.textContent = item.text;

      const score = document.createElement("span");
      score.className = "history-score";
      score.textContent = `${Math.round(item.confidence * 100)}%`;

      li.append(face, quote, score);
      li.addEventListener("click", () => {
        ui.textInput.value = item.text;
        ui.charCount.textContent = item.text.length;
        syncButtonState();
        ui.textInput.focus();
      });
      ui.historyList.appendChild(li);
    }
    ui.historySection.hidden = state.history.length === 0;
  }

  function showError(message) {
    ui.errorMsg.textContent = message;
    ui.errorMsg.hidden = false;
  }

  function hideError() {
    ui.errorMsg.hidden = true;
  }

  function capitalize(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  checkHealth();
})();
