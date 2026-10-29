async (page) => {
  // Chạy bằng Playwright MCP trên một phiên web đã có kết quả và bản render.
  // Không thêm dependency, không gọi Gemini, không sửa dữ liệu của phiên.
  const results = [];
  const initialViewport = page.viewportSize();
  const initialTheme = await page.getByRole('button', { name: /^Giao diện:/ }).getAttribute('aria-label');
  const themeName = initialTheme.replace('Giao diện: ', '');
  const zoomFactor = initialViewport.width / await page.evaluate(() => innerWidth);
  const check = async (state) => {
    const layout = await page.evaluate(() => {
      const main = document.querySelector('main');
      const header = document.querySelector('header');
      const dialog = document.querySelector('[role="dialog"]');
      return {
        viewport: innerWidth,
        document: document.documentElement.scrollWidth,
        main: main.clientWidth,
        content: main.scrollWidth,
        header: header.scrollWidth,
        pageHeight: document.documentElement.scrollHeight,
        viewportHeight: innerHeight,
        dialog: dialog ? { width: dialog.clientWidth, content: dialog.scrollWidth } : null,
      };
    });
    if (layout.document > layout.viewport + 1 || layout.content > layout.main + 1 ||
        layout.header > layout.viewport + 1 || layout.pageHeight > layout.viewportHeight + 1 ||
        (layout.dialog && layout.dialog.content > layout.dialog.width + 1)) {
      throw new Error(`${state}: horizontal overflow ${JSON.stringify(layout)}`);
    }
    results.push({ state, ...layout });
  };
  try {
    for (const theme of ['Sáng', 'Tối']) {
      await page.getByRole('button', { name: /^Giao diện:/ }).click();
      await page.getByRole('menuitemradio', { name: theme, exact: true }).click();
      for (const width of [260, 320, 390, 768, 1366]) {
        await page.setViewportSize({ width: Math.round(width * zoomFactor), height: Math.round(844 * zoomFactor) });
        await check(`${theme}/${width}/result`);
        const json = page.getByRole('button', { name: /Nâng cao: dữ liệu kỹ thuật/ });
        await json.click();
        await check(`${theme}/${width}/json`);
        await json.click();
        await page.getByRole('button', { name: 'Phóng to', exact: true }).click();
        await page.getByRole('dialog').waitFor();
        await check(`${theme}/${width}/zoom`);
        await page.getByRole('radio', { name: '200%', exact: true }).click();
        await check(`${theme}/${width}/zoom200`);
        await page.getByRole('button', { name: 'Đóng', exact: true }).click();
        await page.getByRole('dialog').waitFor({ state: 'hidden' });
      }
    }
    return results;
  } finally {
    await page.setViewportSize(initialViewport);
    await page.getByRole('button', { name: /^Giao diện:/ }).click();
    await page.getByRole('menuitemradio', { name: themeName, exact: true }).click();
  }
}
