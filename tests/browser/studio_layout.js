async (page) => {
  // Chạy trên phiên đã hoàn tất: kiểm tra responsive, theme, log và thanh phase sticky.
  const results = [];
  const initialViewport = page.viewportSize();
  const initialTheme = (await page.getByRole('button', { name: /^Giao diện:/ }).getAttribute('aria-label')).replace('Giao diện: ', '');
  const scale = initialViewport.width / await page.evaluate(() => innerWidth);
  try {
    for (const theme of ['Sáng', 'Tối']) {
      await page.getByRole('button', { name: /^Giao diện:/ }).click();
      await page.getByRole('menuitemradio', { name: theme, exact: true }).click();
      for (const width of [320, 390, 768, 1024, 1366]) {
        await page.setViewportSize({ width: Math.round(width * scale), height: Math.round(844 * scale) });
        for (const phase of ['01 Phân tích', '02 Xử lý', '03 Kết quả']) {
          await page.getByRole('button', { name: phase, exact: true }).click();
          const layout = await page.evaluate(() => {
            const main = document.querySelector('main');
            const workspace = document.querySelector('.studio-workspace');
            const content = document.querySelector('.studio-content').getBoundingClientRect();
            const log = document.querySelector('.studio-log').getBoundingClientRect();
            return { width: innerWidth, document: document.documentElement.scrollWidth,
              main: main.clientWidth, scroll: main.scrollWidth, height: innerHeight,
              documentHeight: document.documentElement.scrollHeight,
              columns: getComputedStyle(workspace).gridTemplateColumns.split(' ').length,
              overlaps: content.right > log.left + 1 && content.bottom > log.top + 1,
            };
          });
          if (layout.document > layout.width + 1 || layout.scroll > layout.main + 1 ||
              layout.documentHeight > layout.height + 1 || layout.overlaps) {
            throw new Error(`${theme}/${width}/${phase}: ${JSON.stringify(layout)}`);
          }
          await page.evaluate(() => { document.querySelector('main').scrollTop = 500; });
          const nav = await page.evaluate(() => ({
            top: document.querySelector('.studio-phase-nav').getBoundingClientRect().top,
            mainTop: document.querySelector('main').getBoundingClientRect().top,
          }));
          if (nav.top < nav.mainTop - 1) throw new Error(`Phase navigation scrolled away: ${JSON.stringify(nav)}`);
          results.push({ theme, phase, ...layout });
        }
      }
    }
    return results;
  } finally {
    await page.setViewportSize(initialViewport);
    await page.getByRole('button', { name: /^Giao diện:/ }).click();
    await page.getByRole('menuitemradio', { name: initialTheme, exact: true }).click();
    await page.getByRole('button', { name: '03 Kết quả', exact: true }).click();
  }
}
