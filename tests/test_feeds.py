"""Feed parsing and fetching tests against synthetic RSS (no network)."""

import httpx

from pythia.feeds import fetch_all, parse_feed, strip_html

RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test Wire</title>
  <item>
    <title>Central bank &amp; markets: rates held</title>
    <link>https://example.com/rates</link>
    <description>&lt;p&gt;The bank &lt;b&gt;held&lt;/b&gt; rates steady.&lt;/p&gt;</description>
    <pubDate>Wed, 19 Aug 2026 09:30:00 GMT</pubDate>
  </item>
  <item>
    <title>No link entry</title>
  </item>
  <item>
    <title>Bad scheme</title>
    <link>javascript:alert(1)</link>
  </item>
  <item>
    <title></title>
    <link>https://example.com/empty-title</link>
  </item>
</channel></rss>
"""


def test_parse_feed_normalizes_entries() -> None:
    articles = parse_feed("https://example.com/rss", RSS_FIXTURE)
    assert len(articles) == 1
    a = articles[0]
    assert a["source"] == "Test Wire"
    assert a["title"] == "Central bank & markets: rates held"
    assert a["url"] == "https://example.com/rates"
    assert a["summary"] == "The bank held rates steady."
    assert a["published_at"].startswith("2026-08-19T09:30:00")


def test_parse_feed_drops_linkless_and_unsafe_entries() -> None:
    articles = parse_feed("https://example.com/rss", RSS_FIXTURE)
    titles = {a["title"] for a in articles}
    assert "No link entry" not in titles
    assert "Bad scheme" not in titles


def test_parse_feed_handles_garbage_bytes() -> None:
    assert parse_feed("https://example.com/rss", b"\x00\x01 not xml at all") == []


def test_strip_html_removes_scripts_and_collapses_whitespace() -> None:
    raw = "<p>Hello  <script>alert(1)</script> <b>world</b></p>\n\n twice"
    assert strip_html(raw) == "Hello world twice"


def test_strip_html_plain_text_passthrough_with_entities() -> None:
    assert strip_html("A &amp; B") == "A & B"


def _rss_with_items(items_xml: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel><title>Bulk Feed</title>'
        f"{items_xml}</channel></rss>"
    ).encode()


def test_parse_feed_caps_entries_per_feed() -> None:
    items = "".join(
        f"<item><title>Item {i}</title><link>https://example.com/{i}</link></item>"
        for i in range(60)
    )
    articles = parse_feed("https://example.com/rss", _rss_with_items(items))
    assert len(articles) == 25


def test_parse_feed_google_news_extracts_publisher() -> None:
    items = (
        "<item><title>Rates held steady - Reuters</title>"
        "<link>https://news.google.com/articles/1</link></item>"
        "<item><title>Plain headline without separator</title>"
        "<link>https://news.google.com/articles/2</link></item>"
    )
    url = "https://news.google.com/rss/search?q=rates"
    articles = parse_feed(url, _rss_with_items(items))
    assert articles[0]["title"] == "Rates held steady"
    assert articles[0]["source"] == "Reuters"
    assert articles[1]["title"] == "Plain headline without separator"
    assert articles[1]["source"] == "Google News"


def test_parse_feed_non_google_titles_with_hyphens_untouched() -> None:
    items = (
        "<item><title>Supply - demand gap widens</title>"
        "<link>https://example.com/a</link></item>"
    )
    articles = parse_feed("https://example.com/rss", _rss_with_items(items))
    assert articles[0]["title"] == "Supply - demand gap widens"
    assert articles[0]["source"] == "Bulk Feed"


def test_parse_feed_applies_friendly_source_label() -> None:
    items = "<item><title>Bank Rate held</title><link>https://x.example/a</link></item>"
    articles = parse_feed("https://www.bankofengland.co.uk/rss/news", _rss_with_items(items))
    assert articles[0]["source"] == "Bank of England"  # feed title says just "News"


def test_fetch_all_contains_per_feed_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "good.example":
            return httpx.Response(200, content=RSS_FIXTURE)
        return httpx.Response(500, text="upstream broken")

    articles, errors = fetch_all(
        ["https://good.example/rss", "https://bad.example/rss"],
        transport=httpx.MockTransport(handler),
    )
    assert [a["url"] for a in articles] == ["https://example.com/rates"]
    assert list(errors) == ["https://bad.example/rss"]
    assert "HTTPStatusError" in errors["https://bad.example/rss"]


def test_fetch_all_all_feeds_down_returns_empty_not_raise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    articles, errors = fetch_all(
        ["https://a.example/rss", "https://b.example/rss"],
        transport=httpx.MockTransport(handler),
    )
    assert articles == []
    assert set(errors) == {"https://a.example/rss", "https://b.example/rss"}
