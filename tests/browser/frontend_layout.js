async (page) => {
  // Kiểm tra màu thực sự render và bố cục ở cả hai tab, hai chế độ màu.
  const results = [];
  for (const colorScheme of ['dark', 'light']) {
    await page.emulateMedia({ colorScheme });
    for (const width of [1366, 995, 768, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 768 });
      for (const name of ['Xử lý ảnh', 'Trò chuyện & tinh chỉnh']) {
        await page.getByRole('tab', { name, exact: true }).click();
        await page.evaluate(() => scrollTo(0, 0));
        const result = await page.evaluate(() => {
          const app = document.querySelector('#doctor-app');
          const panel = [...app.querySelectorAll('[role="tabpanel"]')].find(el => el.getBoundingClientRect().height > 0);
          const rgb = value => value.match(/[\d.]+/g).slice(0, 3).map(Number);
          const luminance = value => rgb(value).map(v => {
            v /= 255;
            return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
          }).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
          const background = el => {
            while (el) {
              const color = getComputedStyle(el).backgroundColor;
              if (color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent') return color;
              el = el.parentElement;
            }
            return 'rgb(255, 255, 255)';
          };
          const text = [app.querySelector('h1'), ...panel.querySelectorAll('h3, p')]
            .filter(el => el.getBoundingClientRect().height > 0);
          const contrasts = text.map(el => {
            const fg = luminance(getComputedStyle(el).color);
            const bg = luminance(background(el));
            return {text: el.textContent.slice(0, 60), ratio: (Math.max(fg, bg) + .05) / (Math.min(fg, bg) + .05)};
          });
          const cards = [...panel.querySelectorAll('.doctor-card')].map(el => el.getBoundingClientRect().toJSON());
          const run = [...panel.querySelectorAll('button')].find(el => /Phân tích & xử lý|1\. Phân tích & hỏi ý/.test(el.textContent));
          return { viewport: innerWidth, overflow: document.documentElement.scrollWidth > innerWidth,
            appWidth: app.getBoundingClientRect().width, contrasts, cards, runBottom: run.getBoundingClientRect().bottom };
        });
        if (result.overflow) throw new Error(`${colorScheme}/${width}/${name}: horizontal overflow`);
        for (const text of result.contrasts) {
          if (text.ratio < 4.5) throw new Error(`${colorScheme}/${width}/${name}: unreadable text ${JSON.stringify(text)}`);
        }
        if (width >= 995 && result.runBottom > 768) throw new Error(`${width}/${name}: primary action below fold (${result.runBottom})`);
        if (width >= 995 && result.appWidth < width - 120) throw new Error(`${width}/${name}: unnecessarily narrow layout (${result.appWidth})`);
        if (width === 390 && result.cards[1].top < result.cards[0].bottom) throw new Error(`${name}: mobile cards must stack`);
        results.push({colorScheme, width, tab: name, minContrast: Math.min(...result.contrasts.map(t => t.ratio)), runBottom: result.runBottom});
      }
    }
  }
  await page.emulateMedia({ colorScheme: 'light' });
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.getByRole('tab', { name: 'Xử lý ảnh', exact: true }).click();
  return results;
}
