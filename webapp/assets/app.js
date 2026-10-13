/* Slaying OOMs 中文版：Markdown 渲染、原文锚点与侧栏目录。 */
(() => {
  const data = document.getElementById('chapter-data');
  const content = document.getElementById('content');
  if (!data || !content) return;
  content.innerHTML = marked.parse(JSON.parse(data.textContent));

  const toc = document.getElementById('toc');
  const headings = [...content.querySelectorAll('h2, h3')];
  const list = document.createElement('ol');
  headings.forEach((heading, i) => {
    const previous = heading.previousElementSibling;
    const anchor = previous?.matches('a[id]') ? previous : previous?.querySelector(':scope > a[id]');
    heading.id = anchor ? anchor.id : `section-${i}`;
    if (anchor) {
      anchor.remove();
      if (previous !== anchor && !previous.textContent.trim() && !previous.children.length) previous.remove();
    }
    const item = document.createElement('li');
    item.className = `toc-${heading.tagName.toLowerCase()}`;
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent.replace(/（[^）]*）$/, '');
    item.append(link);
    list.append(item);
  });
  toc.append(list);
  const links = [...list.querySelectorAll('a')];
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const index = headings.indexOf(entry.target);
      links.forEach((link, i) => link.classList.toggle('active', i === index));
    });
  }, { rootMargin: '-10% 0px -75% 0px' });
  headings.forEach((heading) => spy.observe(heading));

  function followHash() {
    if (location.hash) {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      target?.scrollIntoView();
    }
  }
  // Anchors are inserted after Markdown renders; align the initial deep link too.
  window.addEventListener('load', followHash, { once: true });
  window.addEventListener('hashchange', followHash);
})();
