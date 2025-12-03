// Tab functionality

const Tabs = {
  activeInputTab: "json",
  activeResultTab: "table",

  init() {
    // Input tabs
    const tabJson = document.getElementById("tabJson");
    const tabManual = document.getElementById("tabManual");

    if (tabJson && tabManual) {
      tabJson.addEventListener("click", () => this.switchInputTab("json"));
      tabManual.addEventListener("click", () => this.switchInputTab("manual"));
    }

    // Result tabs
    const resultTabTable = document.getElementById("resultTabTable");
    const resultTabJson = document.getElementById("resultTabJson");

    if (resultTabTable && resultTabJson) {
      resultTabTable.addEventListener("click", () =>
        this.switchResultTab("table")
      );
      resultTabJson.addEventListener("click", () =>
        this.switchResultTab("json")
      );
    }
  },

  switchInputTab(tab) {
    this.activeInputTab = tab;
    const tabJson = document.getElementById("tabJson");
    const tabManual = document.getElementById("tabManual");
    const contentJson = document.getElementById("tabContentJson");
    const contentManual = document.getElementById("tabContentManual");

    if (tab === "json") {
      tabJson.classList.add("active");
      tabManual.classList.remove("active");
      contentJson.classList.remove("hidden");
      contentManual.classList.add("hidden");
    } else {
      tabManual.classList.add("active");
      tabJson.classList.remove("active");
      contentManual.classList.remove("hidden");
      contentJson.classList.add("hidden");
    }
  },

  switchResultTab(tab) {
    this.activeResultTab = tab;
    const tabTable = document.getElementById("resultTabTable");
    const tabJson = document.getElementById("resultTabJson");
    const contentTable = document.getElementById("resultContentTable");
    const contentJson = document.getElementById("resultContentJson");

    if (tab === "table") {
      tabTable.classList.add("active");
      tabJson.classList.remove("active");
      contentTable.classList.remove("hidden");
      contentJson.classList.add("hidden");
    } else {
      tabJson.classList.add("active");
      tabTable.classList.remove("active");
      contentJson.classList.remove("hidden");
      contentTable.classList.add("hidden");
    }
  },
};
