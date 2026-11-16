"""Persisted cache snapshots: pagination, restart, stale-cache, empty-cache hold, migration 4."""

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jo_watchdog.app.caches import FORMAT_VERSION, CacheManager
from jo_watchdog.app.config import Config
from jo_watchdog.app.worker import Worker
from jo_watchdog.ledger.db import Ledger
from jo_watchdog.ledger.states import State
from tests.fakes import FakeClassifier, FakeNotion, FakeTelegram, RecordedDirectory, update

JOB_ORDERS = "3d8fb4d5-9e59-8073-a0d0-000b77632947"
PRIYA = "c0000000-0000-4000-8000-000000000001"


def config(**overrides) -> Config:
    values = dict(
        intake_chat_id=-100123,
        intake_topic_id=5,
        ops_chat_id=-100999,
        ops_topic_id=7,
        job_orders_data_source_id=JOB_ORDERS,
        standing_cc_contact_page_id=PRIYA,
    )
    values.update(overrides)
    return Config(**values)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 22, 0, 0, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, hours):
        self.now += timedelta(hours=hours)


class CacheTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.db"
        self.ledger = Ledger(self.path)
        self.server = RecordedDirectory()
        self.clock = Clock()

    async def asyncTearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def manager(self, ledger=None, **overrides) -> CacheManager:
        return CacheManager(
            ledger=ledger or self.ledger,
            directory=self.server,
            config=config(**overrides),
            clock=self.clock,
        )


class MigrationTests(CacheTestCase):
    def test_migration_4_creates_cache_snapshot(self):
        versions = [r[0] for r in self.ledger._conn.execute("SELECT version FROM schema_version")]
        self.assertIn(4, versions)
        cols = [r[1] for r in self.ledger._conn.execute("PRAGMA table_info(cache_snapshot)")]
        self.assertEqual(
            cols, ["id", "kind", "format_version", "payload_json", "item_count", "fetched_at"]
        )
        indexes = [r[1] for r in self.ledger._conn.execute("PRAGMA index_list(cache_snapshot)")]
        self.assertIn("idx_cache_snapshot_kind_fetched", indexes)

    def test_keeps_last_ten_per_kind(self):
        for i in range(12):
            self.ledger.save_cache_snapshot(
                "people", 1, [{"n": i}], 1, f"2026-10-22T00:00:{i:02d}Z"
            )
        self.ledger.save_cache_snapshot("projects", 1, [{"n": 0}], 1, "2026-10-22T00:00:00Z")
        self.assertEqual(self.ledger.cache_snapshot_count("people"), 10)
        self.assertEqual(self.ledger.cache_snapshot_count("projects"), 1)
        self.assertEqual(self.ledger.latest_cache_snapshot("people")["payload"], [{"n": 11}])


class RefreshTests(CacheTestCase):
    async def test_refresh_persists_every_kind(self):
        caches = self.manager()
        self.assertFalse(caches.ready)
        self.assertEqual(await caches.refresh(), [])
        self.assertTrue(caches.ready)

        self.assertEqual(len(caches.people), 13)
        self.assertEqual(len(caches.projects), 12)
        self.assertEqual(caches.departments[0], "TOC")

        snap = self.ledger.latest_cache_snapshot("people")
        self.assertEqual(snap["item_count"], 13)
        self.assertEqual(snap["format_version"], FORMAT_VERSION)
        self.assertEqual(snap["fetched_at"], "2026-10-22T00:00:00Z")

    async def test_resolution_through_manager(self):
        caches = self.manager()
        await caches.refresh()
        people = caches.resolve_people("Ramon / @user_two", None)
        self.assertEqual(len(people.assigned), 2)
        self.assertEqual(people.cc, [PRIYA])
        self.assertEqual(caches.resolve_project("MFG - TOC").code, "MFG")
        self.assertEqual(caches.resolve_department("TO").department, "TOC")
        self.assertEqual(caches.resolve_department("XX").findings[0].code, "unknown_department")

    async def test_snapshot_loads_after_restart_before_live_refresh(self):
        await self.manager().refresh()
        self.server.fail = True
        restarted = self.manager(ledger=Ledger(self.path))
        restarted.load_persisted()
        self.assertTrue(restarted.ready)
        self.assertEqual(len(restarted.people), 13)
        self.assertEqual(self.server.calls, ["people", "projects", "departments"])  # none after

    async def test_failed_refresh_keeps_snapshot_and_alerts_once_after_24h(self):
        caches = self.manager()
        await caches.refresh()
        self.server.fail = True

        self.clock.advance(6)
        self.assertEqual(await caches.refresh(), [])  # 6h old: not stale yet
        self.assertTrue(caches.ready)
        self.assertEqual(self.ledger.cache_snapshot_count("people"), 1)  # failure writes nothing

        self.clock.advance(19)  # 25h since the last success
        stale = await caches.refresh()
        self.assertEqual(sorted(s.kind for s in stale), ["departments", "people", "projects"])
        self.assertEqual(stale[0].fetched_at, "2026-10-22T00:00:00Z")
        self.assertEqual(stale[0].hours_stale, 25.0)
        self.assertIn("503", stale[0].error)

        self.clock.advance(6)
        self.assertEqual(await caches.refresh(), [])  # reported once per stale episode

        self.server.fail = False
        await caches.refresh()
        self.server.fail = True
        self.clock.advance(25)
        self.assertEqual(len(await caches.refresh()), 3)  # a new episode alerts again

    async def test_empty_refresh_is_a_failure_and_not_persisted(self):
        caches = self.manager()
        await caches.refresh()
        self.server.empty_people = True
        self.clock.advance(6)
        await caches.refresh()
        self.assertEqual(len(caches.people), 13)
        self.assertEqual(self.ledger.cache_snapshot_count("people"), 1)

    async def test_refresh_schedule(self):
        caches = self.manager(cache_refresh_hours=6)
        self.assertTrue(caches.refresh_due())
        await caches.refresh_if_due()
        calls = len(self.server.calls)
        self.clock.advance(5)
        await caches.refresh_if_due()
        self.assertEqual(len(self.server.calls), calls)
        self.clock.advance(1)
        await caches.refresh_if_due()
        self.assertGreater(len(self.server.calls), calls)

    async def test_older_format_snapshot_is_ignored(self):
        self.ledger.save_cache_snapshot(
            "people", FORMAT_VERSION + 1, [{"x": 1}], 1, "2026-10-21T00:00:00Z"
        )
        caches = self.manager()
        caches.load_persisted()
        self.assertIsNone(caches.people)


class EmptyCacheHoldTests(CacheTestCase):
    async def test_never_loaded_cache_holds_messages(self):
        self.server.fail = True
        caches = self.manager()
        telegram = FakeTelegram([update(10, message_id=1, text="MFG - TOC - JO #74")])
        notion_writer = FakeNotion()
        classifier = FakeClassifier()
        worker = Worker(
            ledger=self.ledger,
            telegram=telegram,
            notion=notion_writer,
            job_orders=notion_writer,
            classifier=classifier,
            config=config(),
            caches=caches,
        )
        await worker.run_once()
        self.assertFalse(caches.ready)
        msg = self.ledger.get_message("-100123:1")
        self.assertEqual(msg["state"], State.EXTRACTED.value)
        self.assertEqual(msg["attempts"], 0)
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(notion_writer.created, [])
        self.assertEqual(telegram.reactions, [])

        # Once a snapshot loads, the held message proceeds.
        self.server.fail = False
        self.clock.advance(6)
        await worker.run_once()
        self.assertEqual(self.ledger.get_message("-100123:1")["state"], State.DONE.value)


if __name__ == "__main__":
    unittest.main()
