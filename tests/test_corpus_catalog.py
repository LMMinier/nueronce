from nueronce.corpus.catalog import load_collections, load_works, validate_catalog


def test_expanded_catalog_counts_and_unique_ids():
    collections = load_collections()
    works = load_works()

    assert len(collections) == 153
    assert len({entry.source_id for entry in collections}) == 153
    assert len(works) == 289
    assert len({entry.work_id for entry in works}) == 289


def test_expanded_catalog_validation_summary():
    summary = validate_catalog()

    assert summary["collections"] == 153
    assert summary["works"] == 289
    assert summary["work_shards"] == 6
    assert sum(summary["license_buckets"].values()) == 153
