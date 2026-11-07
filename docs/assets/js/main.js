(() => {
  const nav = document.querySelector(".site-nav");
  const toggle = document.querySelector(".nav-toggle");
  const backdrop = document.querySelector(".nav-backdrop");
  const links = document.querySelectorAll(".nav-links a");
  const sections = [...document.querySelectorAll("main section[id]")];

  const setOpen = (open) => {
    if (!nav || !toggle) return;
    nav.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (backdrop) {
      backdrop.classList.toggle("show", open);
      backdrop.hidden = !open;
    }
    document.body.style.overflow = open ? "hidden" : "";
  };

  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      setOpen(!nav.classList.contains("open"));
    });
  }

  if (backdrop) {
    backdrop.addEventListener("click", () => setOpen(false));
  }

  links.forEach((link) => {
    link.addEventListener("click", () => setOpen(false));
  });

  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("visible"));
  }

  const setActive = () => {
    const y = window.scrollY + 120;
    let current = sections[0]?.id;
    for (const section of sections) {
      if (section.offsetTop <= y) current = section.id;
    }
    links.forEach((link) => {
      const href = link.getAttribute("href") || "";
      link.classList.toggle("active", href === `#${current}`);
    });
  };

  window.addEventListener("scroll", setActive, { passive: true });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 900) setOpen(false);
  });
  setActive();
})();
