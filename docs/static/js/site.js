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
