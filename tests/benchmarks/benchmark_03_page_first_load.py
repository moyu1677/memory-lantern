"""表3：使用Playwright测量登录后页面的FCP和导航时延。"""

from __future__ import annotations

import argparse
import asyncio

from playwright.async_api import async_playwright

from benchmark_common import utc_timestamp, write_csv


DEFAULT_PATHS = ("/dashboard", "/assistant", "/history", "/knowledge")


async def login_and_save_state(browser, base_url, username, password):
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(f"{base_url}/login", wait_until="domcontentloaded")
    await page.locator('input[name="username"]').fill(username)
    await page.locator('input[name="password"]').fill(password)
    await page.locator('button[type="submit"]').click()
    await page.wait_for_url("**/dashboard")
    state = await context.storage_state()
    await context.close()
    return state


async def measure(args):
    rows = []
    base_url = args.base_url.rstrip("/")
    paths = tuple(args.paths or DEFAULT_PATHS)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        state = await login_and_save_state(
            browser,
            base_url,
            args.username,
            args.password,
        )

        for path in paths:
            for run in range(1, args.runs + args.warmups + 1):
                context = await browser.new_context(storage_state=state)
                page = await context.new_page()
                cdp = await context.new_cdp_session(page)
                await cdp.send(
                    "Network.setCacheDisabled",
                    {"cacheDisabled": True},
                )
                response = await page.goto(
                    f"{base_url}{path}",
                    wait_until="networkidle",
                    timeout=args.timeout * 1000,
                )
                metrics = await page.evaluate(
                    """
                    () => {
                        const navigation =
                            performance.getEntriesByType('navigation')[0];
                        const fcp = performance.getEntriesByName(
                            'first-contentful-paint'
                        )[0];
                        return {
                            fcp_ms: fcp ? fcp.startTime : null,
                            navigation_ms: navigation
                                ? navigation.responseEnd - navigation.startTime
                                : null
                        };
                    }
                    """
                )
                if run > args.warmups:
                    rows.append({
                        "timestamp": utc_timestamp(),
                        "path": path,
                        "run": run - args.warmups,
                        "http_status": response.status if response else None,
                        "fcp_ms": metrics["fcp_ms"],
                        "navigation_ms": metrics["navigation_ms"],
                    })
                await context.close()
        await browser.close()

    output = write_csv(
        "page_first_load_results.csv",
        rows,
        [
            "timestamp",
            "path",
            "run",
            "http_status",
            "fcp_ms",
            "navigation_ms",
        ],
    )
    print(f"完成：{len(rows)}条页面测量，结果写入 {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--paths", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(measure(parse_args()))

