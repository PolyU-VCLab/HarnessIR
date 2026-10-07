(function () {
  /* BibTeX copy */
  const copyBtn = document.getElementById("copy-bibtex");
  const bibtexEl = document.getElementById("bibtex-content");

  if (copyBtn && bibtexEl) {
    copyBtn.addEventListener("click", async function () {
      const text = bibtexEl.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copied!";
        copyBtn.classList.add("copied");
        setTimeout(function () {
          copyBtn.textContent = "Copy";
          copyBtn.classList.remove("copied");
        }, 2000);
      } catch (err) {
        const range = document.createRange();
        range.selectNodeContents(bibtexEl);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("copy");
        copyBtn.textContent = "Copied!";
        setTimeout(function () {
          copyBtn.textContent = "Copy";
        }, 2000);
      }
    });
  }

  /* Floating nav: scroll spy */
  const navLinks = document.querySelectorAll(".float-nav a[data-section]");
  const sections = [];

  navLinks.forEach(function (link) {
    const id = link.getAttribute("data-section");
    const el = document.getElementById(id);
    if (el) sections.push({ id: id, el: el, link: link });
  });

  function setActive(id) {
    navLinks.forEach(function (link) {
      link.classList.toggle("active", link.getAttribute("data-section") === id);
    });
  }

  function onScroll() {
    const offset = window.innerHeight * 0.35;
    let current = sections[0] ? sections[0].id : null;

    for (let i = 0; i < sections.length; i++) {
      const rect = sections[i].el.getBoundingClientRect();
      if (rect.top <= offset) {
        current = sections[i].id;
      }
    }
    if (current) setActive(current);
  }

  if (sections.length) {
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
})();
