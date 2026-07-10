from nueronce.ingestion import IngestionCrawler, PolicyGate, score_quality


def _meta(**over):
    base = {
        "robots_status": "allowed",
        "terms_status": "approved",
        "commercial_use": "allowed",
        "pii_risk": 0.0,
        "source_type": "journal_article",
    }
    base.update(over)
    return base


def test_policy_gate_allows_clean_source():
    ok, reason = PolicyGate().allow("https://x", _meta())
    assert ok and reason == "ok"


def test_policy_gate_blocks():
    g = PolicyGate()
    assert g.allow("u", _meta(robots_status="blocked"))[1] == "robots_block"
    assert g.allow("u", _meta(terms_status="blocked"))[1] == "terms_block"
    assert g.allow("u", _meta(commercial_use="prohibited"))[1] == "license_block"
    assert g.allow("u", _meta(pii_risk=0.5))[1] == "pii_risk"


def test_score_quality_bounds():
    low = score_quality(_meta(spam_risk=1.0))
    high = score_quality(
        _meta(
            review_status="peer_reviewed",
            publisher_verified=True,
            citations_present=True,
            structured_sections=True,
            spam_risk=0.0,
        )
    )
    assert 0.0 <= low <= high <= 1.0
    assert high == 1.0


class _Raw:
    def __init__(self, body, meta):
        self.body = body
        self.meta = meta


class _Fetcher:
    def __init__(self, raw):
        self._raw = raw

    def fetch(self, url):
        return self._raw


class _Vault:
    def __init__(self):
        self.stored = []
        self.rejected = []

    def store_raw(self, record, raw):
        self.stored.append(record)

    def record_rejection(self, url, reason, meta):
        self.rejected.append((url, reason))


def test_crawler_stores_allowed_and_rejects_blocked():
    vault = _Vault()
    allowed = IngestionCrawler(PolicyGate(), vault, _Fetcher(_Raw(b"hello", _meta())))
    rec = allowed.ingest_url("https://ok")
    assert rec is not None
    assert rec.source_id.startswith("sha256:")
    assert vault.stored and not vault.rejected

    vault2 = _Vault()
    blocked = IngestionCrawler(
        PolicyGate(), vault2, _Fetcher(_Raw(b"x", _meta(robots_status="blocked")))
    )
    assert blocked.ingest_url("https://no") is None
    assert vault2.rejected and not vault2.stored
