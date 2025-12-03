// Accordion functionality

const Accordion = {
  init() {
    const triggers = document.querySelectorAll(".accordion-trigger");

    triggers.forEach((trigger) => {
      trigger.addEventListener("click", () => {
        const item = trigger.closest(".accordion-item");
        const content = item.querySelector(".accordion-content");
        const isOpen = !content.classList.contains("hidden");

        // Toggle current
        if (isOpen) {
          content.classList.add("hidden");
        } else {
          content.classList.remove("hidden");
        }
      });
    });
  },

  openFirst() {
    const firstItem = document.querySelector(".accordion-item");
    if (firstItem) {
      const content = firstItem.querySelector(".accordion-content");
      if (content) {
        content.classList.remove("hidden");
      }
    }
  },
};
