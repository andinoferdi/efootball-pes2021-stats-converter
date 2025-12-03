// Utility functions

const Utils = {
  // Escape HTML to prevent XSS
  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  },

  // Validate JSON string
  validateJSON(text) {
    if (!text.trim()) {
      return { valid: false, error: "JSON cannot be empty" };
    }
    try {
      const parsed = JSON.parse(text);
      if (
        typeof parsed !== "object" ||
        parsed === null ||
        Array.isArray(parsed)
      ) {
        return { valid: false, error: "JSON must be an object" };
      }
      return { valid: true, data: parsed };
    } catch (e) {
      return { valid: false, error: "Invalid JSON format: " + e.message };
    }
  },

  // Debounce function
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // Animate number counting
  animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const update = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const current = Math.floor(start + (end - start) * easeOutQuart);
      element.textContent = current;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    };
    requestAnimationFrame(update);
  },

  // Animate ring progress
  animateRing(element, percentage) {
    const circumference = 2 * Math.PI * 45; // radius = 45
    const offset = circumference - (percentage / 100) * circumference;
    element.style.strokeDashoffset = offset;
  },

  // Copy to clipboard
  async copyToClipboard(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      button.classList.add("copied");
      const originalHTML = button.innerHTML;
      button.innerHTML = `
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                </svg>
            `;
      setTimeout(() => {
        button.classList.remove("copied");
        button.innerHTML = originalHTML;
      }, 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  },
};
