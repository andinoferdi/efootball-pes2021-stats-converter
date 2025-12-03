// Form handling

const Form = {
  elements: {},
  Utils: {
    // Declare Utils object here
    debounce: (func, wait) => {
      let timeout;
      return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
      };
    },
    validateJSON: (text) => {
      try {
        const data = JSON.parse(text);
        return { valid: true, data: data };
      } catch (e) {
        return { valid: false, error: e.message };
      }
    },
    copyToClipboard: (text, button) => {
      navigator.clipboard
        .writeText(text)
        .then(() => {
          button.textContent = "Copied!";
          setTimeout(() => {
            button.textContent = "Copy";
          }, 2000);
        })
        .catch((err) => {
          console.error("Failed to copy text: ", err);
        });
    },
    animateValue: (element, start, end, duration) => {
      let startTimestamp = null;
      const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        element.textContent = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
          window.requestAnimationFrame(step);
        }
      };
      window.requestAnimationFrame(step);
    },
    animateRing: (element, rating) => {
      // Placeholder for ring animation logic
    },
    escapeHtml: (unsafe) =>
      unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;"),
  },

  init() {
    this.cacheElements();
    this.bindEvents();
  },

  cacheElements() {
    this.elements = {
      form: document.getElementById("convertForm"),
      efJson: document.getElementById("efJson"),
      jsonError: document.getElementById("jsonError"),
      convertBtn: document.getElementById("convertBtn"),
      loading: document.getElementById("loading"),
      error: document.getElementById("error"),
      results: document.getElementById("results"),
      empty: document.getElementById("empty"),
      jsonResult: document.getElementById("jsonResult"),
      markdownTable: document.getElementById("markdownTable"),
      pesOverall: document.getElementById("pesOverall"),
      copyJsonBtn: document.getElementById("copyJsonBtn"),
      overallRingProgress: document.querySelector(".overall-ring-progress"),
    };
  },

  bindEvents() {
    const { form, efJson, copyJsonBtn } = this.elements;

    if (efJson) {
      efJson.addEventListener(
        "input",
        this.Utils.debounce(() => {
          this.validateJsonInput();
        }, 300)
      );
    }

    if (form) {
      form.addEventListener("submit", (e) => this.handleSubmit(e));
    }

    if (copyJsonBtn) {
      copyJsonBtn.addEventListener("click", () => {
        this.Utils.copyToClipboard(
          this.elements.jsonResult.textContent,
          copyJsonBtn
        );
      });
    }
  },

  validateJsonInput() {
    const { efJson, jsonError } = this.elements;
    const text = efJson.value;

    if (!text.trim()) {
      jsonError.textContent = "";
      jsonError.className = "form-hint";
      return;
    }

    const validation = this.Utils.validateJSON(text);
    jsonError.textContent = validation.valid
      ? "Valid JSON format"
      : validation.error;
    jsonError.className = `form-hint ${validation.valid ? "success" : "error"}`;
  },

  collectManualInputs() {
    const efKeys = [
      "offensive_awareness",
      "ball_control",
      "dribbling",
      "tight_possession",
      "low_pass",
      "lofted_pass",
      "finishing",
      "heading",
      "place_kicking",
      "curl",
      "speed",
      "acceleration",
      "kicking_power",
      "jump",
      "physical_contact",
      "balance",
      "stamina",
      "defensive_awareness",
      "defensive_engagement",
      "tackling",
      "aggression",
      "goalkeeping",
      "gk_catching",
      "gk_parrying",
      "gk_reflexes",
      "gk_reach",
      "weak_foot_usage",
      "weak_foot_acc",
      "form",
      "injury_resistance",
    ];

    const stats = {};

    for (const key of efKeys) {
      const input = document.getElementById(key);
      if (input) {
        const value = input.value.trim();
        if (value !== "") {
          const numValue = Number.parseFloat(value);
          if (!isNaN(numValue) && numValue >= 0) {
            stats[key] = numValue;
          }
        }
      }
    }

    return stats;
  },

  showLoading() {
    const { loading, error, results, empty, convertBtn } = this.elements;
    loading.classList.remove("hidden");
    error.classList.add("hidden");
    results.classList.add("hidden");
    empty.classList.add("hidden");
    convertBtn.disabled = true;
    convertBtn.classList.add("loading");
  },

  hideLoading() {
    const { loading, convertBtn } = this.elements;
    loading.classList.add("hidden");
    convertBtn.disabled = false;
    convertBtn.classList.remove("loading");
  },

  showError(message) {
    this.hideLoading();
    const { error, results, empty } = this.elements;
    error.textContent = message;
    error.classList.remove("hidden");
    results.classList.add("hidden");
    empty.classList.add("hidden");
  },

  showResults(pesStats, markdownTableText, overallRating) {
    this.hideLoading();
    const {
      error,
      empty,
      results,
      pesOverall,
      jsonResult,
      overallRingProgress,
    } = this.elements;

    error.classList.add("hidden");
    empty.classList.add("hidden");
    results.classList.remove("hidden");
    results.classList.add("animate-scale-in");

    // Animate overall rating
    if (overallRating !== undefined && overallRating !== null) {
      this.Utils.animateValue(pesOverall, 0, overallRating, 800);
      this.Utils.animateRing(overallRingProgress, overallRating);
    } else {
      pesOverall.textContent = "-";
    }

    // Display JSON
    jsonResult.textContent = JSON.stringify(pesStats, null, 2);

    // Render table
    this.renderMarkdownTable(markdownTableText);
  },

  renderMarkdownTable(markdownText) {
    const { markdownTable } = this.elements;
    const lines = markdownText.split("\n");

    if (lines.length < 2) {
      markdownTable.innerHTML =
        '<p class="text-muted p-4">Table not available</p>';
      return;
    }

    let html = "<table>";
    let isFirstLine = true;
    let rowIndex = 0;

    for (const line of lines) {
      if (!line.trim()) continue;
      if (line.includes("---")) continue;

      if (isFirstLine) {
        html += "<thead><tr>";
        const headers = line
          .split("|")
          .filter((cell) => cell.trim())
          .map((cell) => cell.trim());
        for (const header of headers) {
          html += `<th>${this.Utils.escapeHtml(header)}</th>`;
        }
        html += "</tr></thead><tbody>";
        isFirstLine = false;
        continue;
      }

      const cells = line
        .split("|")
        .filter((cell) => cell.trim())
        .map((cell) => cell.trim());
      if (cells.length > 0) {
        html += `<tr class="stagger-item" style="animation-delay: ${
          rowIndex * 30
        }ms">`;
        for (const cell of cells) {
          html += `<td>${this.Utils.escapeHtml(cell)}</td>`;
        }
        html += "</tr>";
        rowIndex++;
      }
    }

    html += "</tbody></table>";
    markdownTable.innerHTML = html;
  },

  async handleSubmit(e) {
    e.preventDefault();

    let efStats = {};

    if (Tabs.activeInputTab === "json") {
      const text = this.elements.efJson.value.trim();
      const validation = this.Utils.validateJSON(text);

      if (!validation.valid) {
        this.showError(validation.error);
        return;
      }
      efStats = validation.data;
    } else {
      efStats = this.collectManualInputs();

      if (Object.keys(efStats).length === 0) {
        this.showError("Please fill in at least one stat for conversion");
        return;
      }
    }

    const position = document.getElementById("position").value;
    const efOverallInput = document.getElementById("efOverall");
    const efOverall = efOverallInput.value.trim()
      ? Number.parseFloat(efOverallInput.value)
      : null;

    this.showLoading();

    try {
      const requestBody = {
        ef_stats: efStats,
        position: position,
      };

      if (efOverall !== null && !isNaN(efOverall)) {
        requestBody.ef_overall = efOverall;
      }

      const response = await fetch("/api/convert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (!response.ok) {
        this.showError(data.error || "An error occurred during conversion");
        return;
      }

      this.showResults(data.pes_stats, data.markdown_table, data.pes_overall);
    } catch (err) {
      this.showError("An error occurred: " + err.message);
    }
  },
};
