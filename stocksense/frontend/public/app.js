const API_BASE = "http://127.0.0.1:8000";

const tickerSelect = document.getElementById("tickerSelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const quickBtn = document.getElementById("quickBtn");
const gaProgressSection = document.getElementById("gaProgressSection");

const latestSignalEl = document.getElementById("latestSignal");
const nextDayDateEl = document.getElementById("nextDayDate");
const nextDayCloseEl = document.getElementById("nextDayClose");
const confidenceEl = document.getElementById("confidence");
const rmseEl = document.getElementById("rmse");
const maeEl = document.getElementById("mae");
const signalTableBody = document.getElementById("signalTableBody");

let priceChart;
let gaChart;
let rsiChart;
let macdChart;
let defaultTickerForLoad = "";

function signalClass(signal) {
  const key = signal.toLowerCase().replace(" ", "-");
  return `signal-${key}`;
}

function toast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${isError ? "error" : ""}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

function setLoading(isLoading, withGA = false) {
  analyzeBtn.disabled = isLoading;
  quickBtn.disabled = isLoading;
  gaProgressSection.classList.toggle("hidden", !(isLoading && withGA));
}

async function loadTickerOptions() {
  if (!tickerSelect) return;

  try {
    const resp = await fetch(`${API_BASE}/api/tickers`);
    if (!resp.ok) return;
    const data = await resp.json();
    const allTickers = Array.isArray(data.tickers) ? data.tickers : [];
    const indian = allTickers.filter((t) => t.endsWith(".NS"));
    const tickers = (indian.length >= 10 ? indian : allTickers).slice(0, 10);

    tickerSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select company";
    tickerSelect.appendChild(placeholder);

    tickers.forEach((ticker) => {
      const opt = document.createElement("option");
      opt.value = ticker;
      opt.textContent = ticker;
      tickerSelect.appendChild(opt);
    });

    defaultTickerForLoad = tickers.length ? tickers[0] : "";

  } catch {
    // Keep dashboard usable even if ticker lookup fails.
  }
}

function animateMetric(el, value, suffix = "", decimals = 2) {
  const duration = 700;
  const start = performance.now();
  const from = parseFloat(el.dataset.value || "0");

  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const current = from + (value - from) * t;
    el.textContent = `${current.toFixed(decimals)}${suffix}`;
    if (t < 1) {
      requestAnimationFrame(frame);
    } else {
      el.dataset.value = String(value);
    }
  }

  requestAnimationFrame(frame);
}

function renderLatestSignal(signal) {
  latestSignalEl.innerHTML = `<span class="badge ${signalClass(signal)}">${signal}</span>`;
}

function destroyCharts() {
  [priceChart, gaChart, rsiChart, macdChart].forEach((chart) => {
    if (chart) chart.destroy();
  });
}

function buildMainChart(data) {
  const ctx = document.getElementById("priceChart");

  const markerPoints = data.predicted_prices.map((y, i) => ({
    x: data.dates[i],
    y,
    signal: data.signals[i],
  }));

  const markerStyles = markerPoints.map((p) => (p.signal.includes("Buy") ? "triangle" : p.signal.includes("Sell") ? "triangle" : "circle"));
  const markerRotation = markerPoints.map((p) => (p.signal.includes("Sell") ? 180 : 0));
  const markerColors = markerPoints.map((p) => {
    if (p.signal === "Strong Buy") return "#00d26a";
    if (p.signal === "Buy") return "#6ff2a8";
    if (p.signal === "Sell") return "#ff9f43";
    if (p.signal === "Strong Sell") return "#ff4d4f";
    return "#95a2bd";
  });

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Actual Price",
          data: data.actual_prices,
          borderColor: "#4f8ef7",
          backgroundColor: "rgba(79, 142, 247, 0.18)",
          tension: 0.25,
          yAxisID: "y",
        },
        {
          label: "Predicted Price",
          data: data.predicted_prices,
          borderColor: "#8dd0ff",
          borderDash: [6, 4],
          tension: 0.25,
          yAxisID: "y1",
        },
        {
          label: "Signals",
          data: markerPoints,
          showLine: false,
          pointStyle: markerStyles,
          pointRotation: markerRotation,
          pointRadius: 5,
          pointBackgroundColor: markerColors,
          borderWidth: 0,
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          ticks: { color: "#9ca9c7" },
          grid: { color: "rgba(146, 169, 215, 0.08)" },
        },
        y: {
          position: "left",
          ticks: { color: "#9ca9c7" },
          grid: { color: "rgba(146, 169, 215, 0.08)" },
        },
        y1: {
          position: "right",
          ticks: { color: "#9ca9c7" },
          grid: { drawOnChartArea: false },
        },
      },
      plugins: {
        legend: {
          labels: { color: "#d9e1f2" },
        },
      },
    },
  });
}

function buildGAChart(data) {
  const ctx = document.getElementById("gaChart");
  const labels = data.ga_history.map((x) => `Gen ${x.generation}`);
  const values = data.ga_history.map((x) => x.best_fitness);

  gaChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Best Fitness",
          data: values,
          borderColor: "#4f8ef7",
          backgroundColor: "rgba(79, 142, 247, 0.14)",
          fill: true,
          tension: 0.22,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
        y: { ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
      },
      plugins: { legend: { labels: { color: "#d9e1f2" } } },
    },
  });
}

function buildRSIChart(data) {
  const ctx = document.getElementById("rsiChart");
  rsiChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "RSI",
          data: data.rsi,
          borderColor: "#6ff2a8",
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
        y: { min: 0, max: 100, ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
      },
      plugins: { legend: { labels: { color: "#d9e1f2" } } },
    },
  });
}

function buildMACDChart(data) {
  const ctx = document.getElementById("macdChart");
  macdChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "MACD Signal",
          data: data.macd,
          borderColor: "#ff9f43",
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
        y: { ticks: { color: "#9ca9c7" }, grid: { color: "rgba(146,169,215,0.08)" } },
      },
      plugins: { legend: { labels: { color: "#d9e1f2" } } },
    },
  });
}

function renderTable(data) {
  signalTableBody.innerHTML = "";

  data.dates.forEach((date, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${date}</td>
      <td>${Number(data.predicted_prices[i]).toFixed(2)}</td>
      <td>${Number(data.rsi[i]).toFixed(2)}</td>
      <td>${Number(data.macd[i]).toFixed(4)}</td>
      <td><span class="badge ${signalClass(data.signals[i])}">${data.signals[i]}</span></td>
    `;
    signalTableBody.appendChild(tr);
  });
}

function renderDashboard(data) {
  renderLatestSignal(data.latest_signal);
  nextDayDateEl.textContent = data.next_trading_date || "-";
  animateMetric(nextDayCloseEl, Number(data.next_day_predicted_close || 0), "", 2);
  animateMetric(confidenceEl, Number(data.confidence || 0), "%", 1);
  animateMetric(rmseEl, Number(data.metrics?.RMSE || 0), "", 3);
  animateMetric(maeEl, Number(data.metrics?.MAE || 0), "", 3);

  destroyCharts();
  buildMainChart(data);
  buildGAChart(data);
  buildRSIChart(data);
  buildMACDChart(data);
  renderTable(data);
}

async function callApi(path, withGA) {
  const ticker = (tickerSelect?.value || "").trim() || defaultTickerForLoad;
  if (!ticker) {
    toast("Please select a company from the dropdown", true);
    return;
  }

  try {
    setLoading(true, withGA);
    const resp = await fetch(`${API_BASE}${path}${encodeURIComponent(ticker)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed with status ${resp.status}`);
    }

    const data = await resp.json();
    renderDashboard(data);
    toast(`Done in ${data.processing_time_seconds}s`);
  } catch (error) {
    toast(error.message || "Unexpected error", true);
  } finally {
    setLoading(false, false);
  }
}

analyzeBtn.addEventListener("click", () => callApi("/api/analyze?ticker=", false));
quickBtn.addEventListener("click", () => callApi("/api/predict?ticker=", false));

// Initial quick load for a usable first paint.
loadTickerOptions().finally(() => {
  if (defaultTickerForLoad) {
    if (tickerSelect) tickerSelect.value = defaultTickerForLoad;
    callApi("/api/predict?ticker=", false);
  }
});
