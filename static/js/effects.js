// Visual effects and initialization

const Effects = {
  init() {
    this.initIntersectionObserver();
    this.initInputFocusEffects();
    this.initParallax();
  },

  // Intersection observer for scroll animations
  initIntersectionObserver() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      }
    );

    document.querySelectorAll(".card").forEach((el) => {
      observer.observe(el);
    });
  },

  // Enhanced focus effects for inputs
  initInputFocusEffects() {
    const inputs = document.querySelectorAll(
      ".form-input, .form-textarea, .form-select"
    );

    inputs.forEach((input) => {
      input.addEventListener("focus", () => {
        input.parentElement?.classList.add("focused");
      });

      input.addEventListener("blur", () => {
        input.parentElement?.classList.remove("focused");
      });
    });
  },

  // Subtle parallax effect on background
  initParallax() {
    let ticking = false;

    window.addEventListener("scroll", () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const scrolled = window.pageYOffset;
          const orbs = document.querySelectorAll(".gradient-orb");

          orbs.forEach((orb, index) => {
            const speed = index === 0 ? 0.3 : 0.2;
            orb.style.transform = `translateY(${scrolled * speed}px)`;
          });

          ticking = false;
        });

        ticking = true;
      }
    });
  },
};

// Initialize theme toggle button
document.addEventListener("DOMContentLoaded", () => {
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      Theme.toggle();
    });
  }
  
  Tabs.init();
  Accordion.init();
  Form.init();
  Effects.init();
});
