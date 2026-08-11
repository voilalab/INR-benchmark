document.documentElement.classList.add("js");

const tabs = [...document.querySelectorAll("[data-tab]")];
const panels = [...document.querySelectorAll(".result-panel")];

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
    panels.forEach((panel) => { panel.hidden = panel.id !== tab.dataset.tab; });
  });
});

const copyButton = document.querySelector("[data-copy-bib]");
copyButton?.addEventListener("click", async () => {
  const status = document.querySelector(".copy-status");
  try {
    await navigator.clipboard.writeText(document.querySelector("#bibtex").textContent.trim());
    status.textContent = "Copied";
    setTimeout(() => { status.textContent = ""; }, 1800);
  } catch {
    status.textContent = "Select and copy";
  }
});

const methodFamilies = {
  FFN: ["INR", "inr"],
  SIREN: ["INR", "inr"],
  WIRE: ["INR", "inr"],
  "GA-Planes": ["Hybrid", "hybrid"],
  "Instant-NGP": ["Hybrid", "hybrid"],
  GSplat: ["Discrete", "discrete"],
  BACON: ["INR", "inr"],
  Grid: ["Grid", "grid"],
};

const benchmarkData = {
  ct: {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "Sparse-view reconstruction across seven chest CT scans. The TV-regularized Grid leads at every matched budget in the compressive regime.",
    scores: {
      FFN: [12.29, 17.16, 23.17, 26.81, 29.75, 28.50],
      SIREN: [12.20, 10.96, 8.09, 8.52, 7.26, 6.82],
      WIRE: [18.75, 20.69, 21.08, 21.52, 21.86, 22.27],
      "GA-Planes": [31.49, 33.76, 33.15, 32.41, 31.91, 29.89],
      "Instant-NGP": [12.34, 19.03, 18.94, 16.64, 18.32, 22.01],
      GSplat: [27.59, 27.57, 28.36, 27.92, 28.20, 27.00],
      BACON: [16.37, 16.30, 12.07, 13.27, 9.87, 5.11],
      Grid: [38.31, 40.35, 41.13, 40.71, 37.76, 31.82],
    },
  },
  "div2k-denoise-005": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "DIV2K denoising with Gaussian noise ε = 0.05. Hybrid models lead at small budgets; the regularized Grid becomes strongest at larger budgets.",
    scores: {
      FFN: [15.96, 21.83, 27.11, 28.39, 27.16, 26.81],
      SIREN: [21.89, 23.89, 21.81, 19.59, 16.91, 14.17],
      WIRE: [21.54, 23.82, 26.49, 27.86, 27.35, 26.63],
      "GA-Planes": [22.01, 24.78, 27.93, 28.75, 27.77, 28.74],
      "Instant-NGP": [15.72, 20.10, 24.07, 27.43, 27.10, 26.29],
      GSplat: [20.97, 21.54, 21.86, 21.97, 21.93, 21.65],
      BACON: [17.54, 21.55, 26.21, 27.71, 27.16, 11.57],
      Grid: [21.88, 23.87, 26.29, 28.46, 29.61, 28.87],
    },
  },
  "div2k-denoise-01": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "DIV2K denoising with stronger Gaussian noise ε = 0.1. The best representation changes with capacity, exposing overfitting and regularization effects.",
    scores: {
      FFN: [15.94, 21.34, 24.35, 22.56, 21.02, 20.81],
      SIREN: [21.83, 23.45, 20.85, 18.19, 15.83, 11.40],
      WIRE: [21.38, 23.20, 24.14, 23.29, 21.48, 20.67],
      "GA-Planes": [21.87, 23.90, 24.73, 22.59, 20.95, 25.14],
      "Instant-NGP": [15.66, 19.83, 22.49, 22.56, 21.10, 20.50],
      GSplat: [20.92, 21.32, 21.76, 21.83, 21.84, 21.42],
      BACON: [17.47, 21.25, 24.10, 22.81, 21.96, 9.80],
      Grid: [21.78, 23.43, 25.00, 25.74, 24.62, 24.95],
    },
  },
  "div2k-sr": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "DIV2K 4× super-resolution. FFN, WIRE, GA-Planes, and Grid are tightly grouped from 100K parameters upward.",
    scores: {
      FFN: [15.16, 19.51, 21.98, 22.51, 22.72, 22.78],
      SIREN: [19.74, 20.78, 18.75, 16.97, 13.92, 9.95],
      WIRE: [19.45, 20.76, 21.89, 22.41, 22.54, 22.61],
      "GA-Planes": [19.71, 20.83, 21.77, 22.24, 22.34, 21.92],
      "Instant-NGP": [14.39, 17.77, 17.71, 14.03, 12.91, 12.29],
      GSplat: [19.07, 19.40, 19.69, 19.84, 19.66, 19.51],
      BACON: [16.50, 19.38, 21.66, 22.34, 22.43, 7.68],
      Grid: [19.70, 20.85, 21.85, 22.37, 22.46, 22.05],
    },
  },
  "div2k-fit": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "Direct fitting on ten DIV2K images. GA-Planes leads under extreme compression; at image-scale budgets the explicit Grid can represent the samples almost exactly.",
    scores: {
      FFN: [15.99, 22.00, 28.88, 35.95, 44.08, 48.26],
      SIREN: [21.98, 23.93, 22.05, 20.35, 15.87, 11.22],
      WIRE: [21.58, 24.08, 27.67, 31.62, 36.09, 39.06],
      "GA-Planes": [22.05, 24.96, 29.81, 35.20, 38.75, 29.32],
      "Instant-NGP": [15.71, 20.18, 24.73, 32.42, 39.21, 63.21],
      GSplat: [20.97, 21.46, 21.90, 22.07, 21.81, 21.62],
      BACON: [17.57, 21.65, 27.19, 32.17, 34.59, 10.35],
      Grid: [21.92, 24.06, 27.29, 33.01, 149.73, 156.53],
    },
  },
  "dragon-occ-fit": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "Direct fitting of the Stanford Dragon occupancy volume. GA-Planes dominates small and medium budgets; Instant-NGP wins at the largest budget.",
    scores: {
      FFN: [8.56, 10.86, 15.55, 19.79, 42.43, 62.75],
      SIREN: [16.72, 23.08, 26.95, 30.24, 25.89, 45.08],
      WIRE: [23.54, 28.01, 26.41, 24.56, 34.53, 46.17],
      "GA-Planes": [24.23, 30.40, 41.45, 45.90, 48.43, 47.42],
      "Instant-NGP": [8.91, 9.64, 12.11, 25.22, 40.71, 76.17],
      GSplat: [null, null, null, null, null, null],
      BACON: [18.28, 20.39, 22.63, 7.63, 5.64, 3.13],
      Grid: [17.61, 19.40, 21.36, 23.76, 28.47, 42.63],
    },
  },
  "dragon-surface-fit": {
    metric: "PSNR", unit: "dB", decimals: 2,
    description: "Direct fitting of the sparse Stanford Dragon surface. Adaptive 3D representations benefit from the signal’s lower-dimensional structure.",
    scores: {
      FFN: [13.42, 14.67, 19.27, 35.04, 73.36, 47.47],
      SIREN: [14.60, 15.92, 18.25, 28.28, 16.27, 38.71],
      WIRE: [15.68, 16.42, 18.81, 23.61, 33.29, 37.08],
      "GA-Planes": [17.99, 22.62, 38.86, 40.86, 39.44, 43.11],
      "Instant-NGP": [13.62, 14.44, 18.81, 45.37, 78.03, 77.92],
      GSplat: [null, null, null, null, null, null],
      BACON: [15.83, 17.77, 20.48, 22.66, 7.60, 3.20],
      Grid: [14.87, 15.83, 17.51, 20.10, 25.35, 40.33],
    },
  },
  "dragon-occ-sr": {
    metric: "IoU", unit: "", decimals: 2,
    description: "Stanford Dragon occupancy super-resolution. GA-Planes maintains about 0.95 IoU across the full capacity sweep.",
    scores: {
      FFN: [0.28, 0.37, 0.28, 0.24, 0.29, 0.35],
      SIREN: [0.75, 0.91, 0.93, 0.82, 0.43, 0.22],
      WIRE: [0.92, 0.94, 0.93, 0.67, 0.75, 0.81],
      "GA-Planes": [0.93, 0.95, 0.95, 0.95, 0.95, 0.95],
      "Instant-NGP": [0.21, 0.30, 0.43, 0.45, 0.43, 0.37],
      GSplat: [null, null, null, null, null, null],
      BACON: [0.79, 0.84, 0.88, 0.16, 0.16, 0.16],
      Grid: [0.77, 0.82, 0.86, 0.88, 0.91, 0.90],
    },
  },
  "dragon-surface-sr": {
    metric: "IoU", unit: "", decimals: 2,
    description: "Stanford Dragon surface super-resolution. GA-Planes is strongest throughout; Grid becomes the second-best method at larger budgets.",
    scores: {
      FFN: [0.00, 0.03, 0.05, 0.07, 0.12, 0.08],
      SIREN: [0.28, 0.36, 0.45, 0.57, 0.27, 0.04],
      WIRE: [0.30, 0.24, 0.17, 0.16, 0.26, 0.47],
      "GA-Planes": [0.44, 0.55, 0.59, 0.60, 0.59, 0.60],
      "Instant-NGP": [0.02, 0.06, 0.12, 0.17, 0.17, 0.22],
      GSplat: [null, null, null, null, null, null],
      BACON: [0.33, 0.40, 0.48, 0.51, 0.04, 0.04],
      Grid: [0.27, 0.32, 0.38, 0.42, 0.50, 0.50],
    },
  },
};

const taskSelect = document.querySelector("[data-result-task]");
const budgetSelect = document.querySelector("[data-result-budget]");
const resultsBody = document.querySelector("[data-results-body]");

function renderBenchmarkTable() {
  if (!taskSelect || !budgetSelect || !resultsBody) return;

  const task = benchmarkData[taskSelect.value];
  const budgetIndex = Number(budgetSelect.value);
  const rows = Object.entries(task.scores)
    .map(([method, values]) => ({ method, score: values[budgetIndex] }))
    .filter(({ score }) => Number.isFinite(score))
    .sort((a, b) => b.score - a.score);
  const winner = rows[0];
  const gridScore = task.scores.Grid[budgetIndex];
  const maxScore = winner.score;

  document.querySelector("[data-result-metric]").textContent = `${task.metric} · higher is better`;
  document.querySelector("[data-result-winner]").textContent = `Winner: ${winner.method} · ${winner.score.toFixed(task.decimals)}${task.unit ? ` ${task.unit}` : ""}`;
  document.querySelector("[data-result-description]").textContent = task.description;
  document.querySelector("[data-score-heading]").textContent = task.unit ? `${task.metric} (${task.unit})` : task.metric;

  resultsBody.replaceChildren(...rows.map(({ method, score }, index) => {
    const row = document.createElement("tr");
    if (index === 0) row.classList.add("row-best");
    if (method === "Grid") row.classList.add("row-grid");
    const delta = score - gridScore;
    const deltaText = method === "Grid" ? "baseline" : `${delta >= 0 ? "+" : ""}${delta.toFixed(task.decimals)}`;
    const [family, familyClass] = methodFamilies[method];
    const barWidth = Math.max(4, (score / maxScore) * 100);
    row.innerHTML = `<td>${index + 1}</td><th>${method}</th><td><span class="family-tag ${familyClass}">${family}</span></td><td>${score.toFixed(task.decimals)}</td><td>${deltaText}</td><td><span class="score-track" aria-label="${Math.round(barWidth)} percent of the best score"><span class="score-fill" style="width:${barWidth}%"></span></span></td>`;
    return row;
  }));
}

taskSelect?.addEventListener("change", renderBenchmarkTable);
budgetSelect?.addEventListener("change", renderBenchmarkTable);
renderBenchmarkTable();
