// Theme management

const Theme = {
  init() {
    const savedTheme = localStorage.getItem("theme");
    const theme = savedTheme || "dark";
    
    this.setTheme(theme, false);
    this.updateToggleIcon(theme);
  },

  getTheme() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  },

  setTheme(theme, save = true) {
    if (theme !== "light" && theme !== "dark") {
      theme = "dark";
    }
    
    document.documentElement.setAttribute("data-theme", theme);
    
    if (save) {
      localStorage.setItem("theme", theme);
    }
    
    this.updateToggleIcon(theme);
  },

  toggle() {
    const currentTheme = this.getTheme();
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    this.setTheme(newTheme);
  },

  updateToggleIcon(theme) {
    const lightIcon = document.getElementById("themeIconLight");
    const darkIcon = document.getElementById("themeIconDark");
    
    if (lightIcon && darkIcon) {
      if (theme === "dark") {
        lightIcon.classList.remove("hidden");
        darkIcon.classList.add("hidden");
      } else {
        lightIcon.classList.add("hidden");
        darkIcon.classList.remove("hidden");
      }
    }
  },
};

