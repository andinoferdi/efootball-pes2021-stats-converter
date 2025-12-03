document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("convertForm");
    const efJsonTextarea = document.getElementById("efJson");
    const jsonError = document.getElementById("jsonError");
    const convertBtn = document.getElementById("convertBtn");
    const loading = document.getElementById("loading");
    const error = document.getElementById("error");
    const results = document.getElementById("results");
    const empty = document.getElementById("empty");
    const jsonResult = document.getElementById("jsonResult");
    const markdownTable = document.getElementById("markdownTable");
    const pesOverall = document.getElementById("pesOverall");
    
    const tabJson = document.getElementById("tabJson");
    const tabManual = document.getElementById("tabManual");
    const tabContentJson = document.getElementById("tabContentJson");
    const tabContentManual = document.getElementById("tabContentManual");
    
    let activeTab = "json";
    
    function switchTab(tab) {
        activeTab = tab;
        
        if (tab === "json") {
            tabJson.classList.add("border-blue-600", "text-blue-600");
            tabJson.classList.remove("border-transparent", "text-gray-500");
            tabManual.classList.add("border-transparent", "text-gray-500");
            tabManual.classList.remove("border-blue-600", "text-blue-600");
            tabContentJson.classList.remove("hidden");
            tabContentManual.classList.add("hidden");
        } else {
            tabManual.classList.add("border-blue-600", "text-blue-600");
            tabManual.classList.remove("border-transparent", "text-gray-500");
            tabJson.classList.add("border-transparent", "text-gray-500");
            tabJson.classList.remove("border-blue-600", "text-blue-600");
            tabContentManual.classList.remove("hidden");
            tabContentJson.classList.add("hidden");
        }
    }
    
    tabJson.addEventListener("click", () => switchTab("json"));
    tabManual.addEventListener("click", () => switchTab("manual"));
    
    function collectManualInputs() {
        const efKeys = [
            "offensive_awareness", "ball_control", "dribbling", "tight_possession",
            "low_pass", "lofted_pass", "finishing", "heading", "place_kicking", "curl",
            "speed", "acceleration", "kicking_power", "jump", "physical_contact",
            "balance", "stamina", "defensive_awareness", "defensive_engagement",
            "tackling", "aggression", "goalkeeping", "gk_catching", "gk_parrying",
            "gk_reflexes", "gk_reach", "weak_foot_usage", "weak_foot_acc",
            "form", "injury_resistance"
        ];
        
        const stats = {};
        
        for (const key of efKeys) {
            const input = document.getElementById(key);
            if (input) {
                const value = input.value.trim();
                if (value !== "") {
                    const numValue = parseFloat(value);
                    if (!isNaN(numValue) && numValue >= 0) {
                        stats[key] = numValue;
                    }
                }
            }
        }
        
        return stats;
    }

    function validateJSON(text) {
        if (!text.trim()) {
            return { valid: false, error: "JSON tidak boleh kosong" };
        }
        try {
            const parsed = JSON.parse(text);
            if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
                return { valid: false, error: "JSON harus berupa object" };
            }
            return { valid: true, data: parsed };
        } catch (e) {
            return { valid: false, error: "Format JSON tidak valid: " + e.message };
        }
    }

    function updateJsonError(valid, message) {
        jsonError.textContent = message || "";
        jsonError.className = valid ? "mt-2 text-xs text-gray-500 success" : "mt-2 text-xs text-gray-500 error";
    }

    efJsonTextarea.addEventListener("input", function() {
        const text = this.value;
        if (!text.trim()) {
            updateJsonError(true, "");
            return;
        }
        const validation = validateJSON(text);
        updateJsonError(validation.valid, validation.valid ? "Format JSON valid" : validation.error);
    });

    function showLoading() {
        loading.classList.remove("hidden");
        error.classList.add("hidden");
        results.classList.add("hidden");
        empty.classList.add("hidden");
        convertBtn.disabled = true;
        convertBtn.textContent = "Memproses...";
    }

    function hideLoading() {
        loading.classList.add("hidden");
        convertBtn.disabled = false;
        convertBtn.textContent = "Konversi";
    }

    function showError(message) {
        hideLoading();
        error.textContent = message;
        error.classList.remove("hidden");
        results.classList.add("hidden");
        empty.classList.add("hidden");
    }

    function showResults(pesStats, markdownTableText, overallRating) {
        hideLoading();
        error.classList.add("hidden");
        empty.classList.add("hidden");
        results.classList.remove("hidden");

        if (overallRating !== undefined && overallRating !== null) {
            pesOverall.textContent = overallRating;
        } else {
            pesOverall.textContent = "-";
        }

        jsonResult.textContent = JSON.stringify(pesStats, null, 2);
        renderMarkdownTable(markdownTableText);
    }

    function renderMarkdownTable(markdownText) {
        const lines = markdownText.split("\n");
        if (lines.length < 2) {
            markdownTable.innerHTML = "<p class='text-gray-500'>Tabel tidak tersedia</p>";
            return;
        }

        let html = "<table class='w-full'>";
        let isFirstLine = true;

        for (const line of lines) {
            if (!line.trim()) continue;

            if (isFirstLine) {
                html += "<thead><tr>";
                const headers = line.split("|").filter(cell => cell.trim()).map(cell => cell.trim());
                for (const header of headers) {
                    html += `<th>${escapeHtml(header)}</th>`;
                }
                html += "</tr></thead><tbody>";
                isFirstLine = false;
                continue;
            }

            if (line.includes("---")) continue;

            const cells = line.split("|").filter(cell => cell.trim()).map(cell => cell.trim());
            if (cells.length > 0) {
                html += "<tr>";
                for (const cell of cells) {
                    html += `<td>${escapeHtml(cell)}</td>`;
                }
                html += "</tr>";
            }
        }

        html += "</tbody></table>";
        markdownTable.innerHTML = html;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        let efStats = {};
        
        if (activeTab === "json") {
            const jsonText = efJsonTextarea.value.trim();
            const validation = validateJSON(jsonText);

            if (!validation.valid) {
                showError(validation.error);
                return;
            }
            
            efStats = validation.data;
        } else {
            efStats = collectManualInputs();
            
            if (Object.keys(efStats).length === 0) {
                showError("Minimal isi satu stat untuk konversi");
                return;
            }
        }

        const position = document.getElementById("position").value;
        const efOverallInput = document.getElementById("efOverall");
        const efOverall = efOverallInput.value.trim() ? parseFloat(efOverallInput.value) : null;

        showLoading();

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
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestBody),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "Terjadi kesalahan saat konversi");
                return;
            }

            showResults(data.pes_stats, data.markdown_table, data.pes_overall);
        } catch (err) {
            showError("Terjadi kesalahan: " + err.message);
        }
    });
});

