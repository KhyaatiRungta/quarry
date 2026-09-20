"""The site is hand-written, so these tests guard the two claims it makes about itself:
that its numbers come from the results file, and that it is standalone."""
import json
import re

import pytest


@pytest.fixture(scope="module")
def page(request):
    root = request.config.rootpath
    return (root / "site/index.html").read_text()


@pytest.fixture(scope="module")
def results(request):
    return json.loads((request.config.rootpath / "results/bench_results.json").read_text())


def test_required_sections_appear_in_order(page):
    headings = re.findall(r"<h2>(.*?)</h2>", page)
    assert headings == [
        "Three things it does that matter",
        "How it works",
        "Benchmarks",
        "A real repair, end to end",
        "Design decisions",
        "Limitations",
        "Running it locally",
    ]


def test_product_page_furniture_is_present(page):
    assert '<nav class="nav">' in page
    assert page.count('class="btn btn-primary"') == 2      # hero and closing band
    assert 'class="btn btn-secondary"' in page
    assert 'class="metrics"' in page
    assert page.count('<div class="card">') == 3
    assert '<div class="band">' in page
    assert 'class="term"' in page


def test_the_hero_headline_is_benefit_led_and_short(page):
    headline = re.search(r"<h1>(.*?)</h1>", page, re.DOTALL).group(1)
    assert len(headline.split()) <= 18
    assert "quarry" not in headline.lower()


def test_no_marketing_superlatives(page):
    banned = ["blazing", "revolutionary", "seamless", "powerful", "cutting-edge",
              "world-class", "effortless", "game-chang"]
    lowered = page.lower()
    for word in banned:
        assert word not in lowered, f"superlative in the copy: {word}"


def test_no_gradients_or_shadows_in_the_stylesheet(request):
    css = (request.config.rootpath / "site/style.css").read_text()
    assert "gradient" not in css
    assert "box-shadow" not in css
    assert "border-radius: 999" not in css


def test_the_metric_strip_is_generated_from_the_results_file(page, results):
    block = page[page.index("<!-- METRICS:BEGIN"):page.index("<!-- METRICS:END")]
    assert str(results["tasks_total"]) in block
    on = results["arms"]["on"]
    assert f"{100 * on['repair_success_rate']:.1f}%" in block
    if results["status"] != "complete":
        assert "pending" in block


def test_the_results_table_is_generated_from_the_results_file(page, results):
    start = page.index("<!-- RESULTS:BEGIN")
    end = page.index("<!-- RESULTS:END")
    block = page[start:end]
    on = results["arms"]["on"]
    off = results["arms"]["off"]
    assert f"{100 * on['task_success_rate']:.1f}%" in block
    assert f"{100 * off['task_success_rate']:.1f}%" in block
    assert f"${on['mean_cost_usd_per_task']:.4f}" in block
    assert results["status"] in block
    assert results["note"] in block


def test_unmeasured_numbers_are_marked_pending_not_invented(page, results):
    if results["status"] != "complete":
        assert "pending" in page
        assert "not yet run" in page


def test_the_page_is_standalone(page):
    """No JavaScript and no remote assets. Outbound links to the repo are fine."""
    assert "<script" not in page
    assert page.count('<link rel="stylesheet" href="style.css">') == 1
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    assert all(url.startswith("https://github.com/") for url in remote), remote


def test_the_footer_links_the_sibling_projects_and_names_nobody(page):
    footer = page[page.index("<footer>"):]
    for repo in ("quarry", "strata", "caliper"):
        assert f"github.com/Manavarya09/{repo}" in footer
    assert "Manav Arya" not in page


def test_no_icons_or_emoji(request):
    root = request.config.rootpath
    for path in ["site/index.html", "site/style.css", "site/trace.html", "README.md"]:
        text = (root / path).read_text()
        assert not re.search(r"[\U0001F000-\U0001FAFF☀-➿]", text), f"emoji in {path}"
        assert "font-awesome" not in text.lower()
        assert "<i class=" not in text


def test_the_architecture_diagram_shows_the_repair_edge(page):
    svg = page[page.index("<svg"):page.index("</svg>")]
    assert "REPAIR" in svg
    assert "traceback" in svg
    assert "verify" in svg
    assert "AST guard" in svg


def test_limitations_are_specific_and_plural(page):
    block = page[page.index('id="limitations"'):page.index('id="run"')]
    items = re.findall(r"<li>", block)
    assert len(items) >= 4
    assert "not a security boundary" in block


def test_the_rendered_trace_is_shipped_and_linked(page, request):
    assert 'href="trace.html"' in page
    trace = (request.config.rootpath / "site/trace.html").read_text()
    assert trace.startswith("<!doctype html>")
    assert "data:image/png;base64," in trace
