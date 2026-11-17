#!/usr/bin/env python3
"""Manual real-account smoke verification (issue #27). Never run in CI.

Checks the assumptions CI deliberately excludes: live Telegram, the t6 API's reads and Job
Order raise, extraction and Deliverable matching through it, the deployment's storage
permissions, and the full capture pipeline against real accounts. Each step prints PASS, FAIL, SKIP, MANUAL or
ATTESTED. The run ends with a verdict, and only FULL VERIFICATION exits 0:

- FULL (exit 0): every required step is PASS or ATTESTED. Only ``live_telegram`` is optional.
- FAILED (exit 1): any step failed.
- INCOMPLETE or PARTIAL (exit 3): a required step was SKIP, MANUAL or not run, or
  ``--only`` ran a subset. A skipped check is never a success.

    python smoke.py --config config.yaml                 # everything except live Telegram
    python smoke.py --config config.yaml --live-telegram --live-message-id 12345
    python smoke.py --config config.yaml --fixture-message-id 900001 --attestations att.json
    python smoke.py --config config.yaml --only reactions,api_access
    python smoke.py --config config.yaml --forward-fixtures fw.json --reaction-message-id 12345

Secrets come from the environment (``.env``): TELEGRAM_BOT_TOKEN, T6_API_URL, T6_API_SECRET.

Side effects, by step:

- ``pipeline`` and ``dry_run`` build the worker with the production wiring
  (``jo_watchdog.app.wiring``: caches, matcher, Deliverable source) on a
  throwaway ledger, with a fixed classifier verdict and a recording Telegram.
  The fixture is one explicit message (``--fixture-message-id``, printed). The
  step verifies that message's classification, plan, Job Order raise and read-back,
  delivered capture report and ``reaction_state = set``, not the last ledger row.
  ``pipeline`` raises one clearly marked Job Order (the API cannot archive it; the
  URL is printed so you can delete it by hand). ``query_visibility`` then asks ``jo find``
  for that message until the Job Order shows.
- ``api_access`` reads People, Projects and Departments through the t6 API and asks
  ``jo find`` once; it fails when the API key cannot read one of them.
- ``extraction`` makes two live ``llm run`` calls (a text with a credential and a text without)
  through the production classifier; ``jev`` makes three live ``llm decide`` calls, one for each
  regression pair (``tests/fixtures/jev/regression_pairs.json``), validated by the production
  matcher. Both need explicit authorization and cost tokens. They print the returned snapshot,
  token counts and probabilities; a refusal prints its class and any ``retry_after``.
- ``threshold_tuning`` is offline: it scores the recorded ``llm decide`` answers, rewrites
  ``docs/tuning/threshold-results.json`` and stays MANUAL until the owner attests the
  review. It never edits config or enables linking.
- ``live_telegram`` (only with ``--live-telegram --live-message-id N``) waits for
  the one intake post with that Telegram message id and ignores every other
  update. Post a test JO in the intake topic first and note its message id. It
  captures into a throwaway ledger (the service's ledger and offset are not
  touched), raises a real Job Order, reacts to the post, posts the capture
  report and one smoke line to the ops topic, then runs the same checks.
  Stop the running watchdog first: two pollers cannot share one bot token.
  Cleanup: delete the Job Order row by hand (URL printed) and the smoke line.

- ``forward_origins`` and ``channel_link`` read ``--forward-fixtures``: a JSON object
  keyed ``user``, ``hidden_user``, ``chat``, ``channel`` whose values are the raw Bot API
  message (or update, as stored in the ledger's ``raw_source``) of a real forward.
  Forward real posts to the bot to capture them. A missing origin is a gap (SKIP); the
  channel link must match the raw chat and message id; every other origin must have none.
- ``reaction_apply`` (needs ``--reaction-message-id``) sets the captured reaction, or the
  flagged one with ``--reaction flagged``, on that real intake message with the production
  client; ``reactions`` rejects either configured reaction missing from ``getChat``'s list
  (or non-standard when it is absent).
- ``owner_handle`` is MANUAL until an attestation whose scope or evidence names both
  configured handles settles it.

Steps the script cannot run report SKIP (or MANUAL). A person who checked one by
hand records it in an ``--attestations`` JSON list of ``{step, actor, at, scope,
evidence}`` (``at`` an ISO-8601 time); all five fields are required, and an
attestation settles only a SKIP or MANUAL step, never a FAIL or PASS.

Checks that are covered by CI but not yet automated against the real accounts
report SKIP with their ticket number; verify those by hand and attest them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from jo_watchdog.adapters.protocols import ClassifyResult, IncomingUpdate
from jo_watchdog.app.config import (
    SECRET_KEYS,
    Config,
    load_config,
    missing_secrets,
    placeholder_keys,
    require_live_keys,
    t6_api_secret,
    t6_api_url,
)
from jo_watchdog.app.model_smoke import ModelCheck, check_decide, check_extraction
from jo_watchdog.app.smoke_gate import (
    FAIL,
    MANUAL,
    PASS,
    SKIP,
    Result,
    attest,
    load_attestations,
    verdict,
    verify_capture,
)
from jo_watchdog.app.smoke_probes import (
    evaluate_channel_link,
    evaluate_forward_origins,
    evaluate_owner_handle,
    evaluate_reactions,
    load_forward_fixtures,
)
from jo_watchdog.app.smoke_scenarios import SCENARIOS
from jo_watchdog.app.tuning import build_report, match_samples, read_report, write_report
from jo_watchdog.app.wiring import build_adapters, build_worker, missing_wiring
from jo_watchdog.core.extraction import EXTRACTED_FIELDS, PROMPT_TAG, Extraction
from jo_watchdog.core.payload import page_url_for
from jo_watchdog.core.report import DRY_RUN_HEADER
from jo_watchdog.ledger.db import Ledger

ROOT = Path(__file__).resolve().parent
SMOKE_MARK = "SMOKE TEST — delete this row"


class RecordingTelegram:
    """Records reactions and ops posts instead of sending them (no token needed)."""

    def __init__(self, updates=()):
        self.updates: list[IncomingUpdate] = list(updates)
        self.reactions: list[tuple[int, int, str]] = []
        self.sent: list[tuple[int, int, str]] = []

    async def get_updates(self, offset: int, timeout: int) -> list[IncomingUpdate]:
        batch, self.updates = [u for u in self.updates if u.update_id >= offset], []
        return batch

    async def set_reaction(self, chat_id: int, message_id: int, emoji: str) -> None:
        self.reactions.append((chat_id, message_id, emoji))

    async def send_message(self, chat_id: int, topic_id: int, text: str) -> int:
        self.sent.append((chat_id, topic_id, text))
        return len(self.sent)


class AlwaysJoClassifier:
    """Stands in for the classifier in the synthetic pipeline steps."""

    async def classify(self, text: str, origin_lines) -> ClassifyResult:
        extraction = Extraction(
            verdict="jo",
            confidence="high",
            reasons="Smoke test stub.",
            multiple_jo_suspected=False,
            **{field: None for field in EXTRACTED_FIELDS},
        )
        return ClassifyResult(
            outcome="ok", extraction=extraction, requested_model="stub", prompt_tag=PROMPT_TAG
        )


def synthetic_update(config: Config, text: str, message_id: int) -> IncomingUpdate:
    return IncomingUpdate(
        update_id=1,
        chat_id=config.intake_chat_id,
        message_id=message_id,
        message_thread_id=config.intake_topic_id,
        text=text,
        raw={"update_id": 1, "smoke": True},
    )


# -- result plumbing ----------------------------------------------------------


def ok(detail: str = "") -> Result:
    return Result(PASS, detail)


def fail(detail: str) -> Result:
    return Result(FAIL, detail)


def skip(detail: str) -> Result:
    return Result(SKIP, detail)


def manual(detail: str) -> Result:
    return Result(MANUAL, detail)


class Smoke:
    """Shared state for one smoke run: config, clients, and facts found so far."""

    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.bot = None
        self.smoke_page_key: str | None = None
        self.attestations: dict = {}
        # One explicit identity for the synthetic fixture, printed so it can be traced.
        self.fixture_message_id = args.fixture_message_id or int(time.time() * 1000) % 2_000_000_000

    async def open(self) -> None:
        if os.environ.get("TELEGRAM_BOT_TOKEN"):
            from aiogram import Bot

            self.bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])

    async def close(self) -> None:
        if self.bot:
            await self.bot.session.close()

# -- steps --------------------------------------------------------------------


async def step_config(s: Smoke) -> Result:
    require_live_keys(s.config)
    placeholders = placeholder_keys(s.config)
    missing = missing_secrets()
    if missing or placeholders:
        return fail(f"missing secrets: {missing or '-'}; placeholder values: {placeholders or '-'}")
    return ok(f"all {len(Config.model_fields)} keys valid; {len(SECRET_KEYS)} secrets present")


async def step_storage(s: Smoke) -> Result:
    ledger = Path(s.config.ledger_path)
    if not ledger.parent.exists():
        return fail(f"{ledger.parent} does not exist; create it with mode 0700")
    problems = []
    dir_mode = stat.S_IMODE(ledger.parent.stat().st_mode)
    if dir_mode & 0o077:
        problems.append(f"{ledger.parent} is {dir_mode:04o}, want 0700")
    for path in sorted(ledger.parent.iterdir()):
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_file() and mode & 0o077:
            problems.append(f"{path.name} is {mode:04o}, want 0600")
    if problems:
        return fail("; ".join(problems))
    return ok(f"{ledger.parent} {dir_mode:04o}, files owner-only")


async def step_reactions(s: Smoke) -> Result:
    if not s.bot:
        return skip("TELEGRAM_BOT_TOKEN not set")
    chat = await s.bot.get_chat(s.config.intake_chat_id)
    verdicts = []
    for name in ("captured_reaction", "flagged_reaction"):
        result, shown = evaluate_reactions(chat.available_reactions, getattr(s.config, name))
        verdicts.append((name, result))
    print(f"    available_reactions: {shown}")
    for name, result in verdicts:
        if result.status != PASS:
            return Result(result.status, f"{name}: {result.detail}")
    return Result(PASS, f"{s.config.captured_reaction} and {s.config.flagged_reaction} are available")


async def step_reaction_apply(s: Smoke) -> Result:
    """Apply a configured reaction (``--reaction`` picks which) to one real intake message."""
    if not s.args.reaction_message_id:
        return skip("--reaction-message-id <id> (a real message in the intake group) is required")
    if missing_secrets():
        return skip(f"production Telegram client needs every secret; missing {missing_secrets()}")
    chat_id, message_id = s.config.intake_chat_id, s.args.reaction_message_id
    emoji = getattr(s.config, f"{s.args.reaction}_reaction")
    adapters = build_adapters(s.config)
    try:
        await adapters.telegram.set_reaction(chat_id, message_id, emoji)
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")
    finally:
        await adapters.close()
    return ok(
        f"Telegram accepted {emoji} on {chat_id}:{message_id}; "
        "look at that message to see it"
    )


async def step_bot_rights(s: Smoke) -> Result:
    if not s.bot:
        return skip("TELEGRAM_BOT_TOKEN not set")
    me = await s.bot.get_me()
    intake = await s.bot.get_chat_member(s.config.intake_chat_id, me.id)
    ops = await s.bot.get_chat_member(s.config.ops_chat_id, me.id)
    problems = []
    if intake.status not in ("administrator", "creator"):
        problems.append(f"not an admin of the intake group ({intake.status})")
    if ops.status in ("left", "kicked") or getattr(ops, "can_send_messages", True) is False:
        problems.append(f"cannot post in the ops group ({ops.status})")
    if problems:
        return fail("; ".join(problems))
    return ok(f"@{me.username}: intake {intake.status}, ops {ops.status}")


async def step_owner_handle(s: Smoke) -> Result:
    return evaluate_owner_handle(
        s.config.owner_telegram_handle,
        s.config.standing_cc_telegram_handle,
        s.attestations.get("owner_handle"),
    )


def _forward_fixtures(s: Smoke) -> dict | None:
    return load_forward_fixtures(s.args.forward_fixtures) if s.args.forward_fixtures else None


async def step_forward_origins(s: Smoke) -> Result:
    return evaluate_forward_origins(_forward_fixtures(s))


async def step_channel_link(s: Smoke) -> Result:
    return evaluate_channel_link(_forward_fixtures(s))


class ReadOnlyApi:
    """The real t6 API for reads and previews; a Job Order raise (confirm) is forbidden."""

    def __init__(self, api):
        self._api = api

    def __getattr__(self, name):
        return getattr(self._api, name)

    async def confirm_job_order(self, inputs, token):
        raise AssertionError("dry run called jo raise with a confirm")

    async def read_job_order(self, page_id):
        raise AssertionError("dry run called jo get")


async def synthetic_capture(s: Smoke, text: str, *, dry_run: bool) -> tuple[Result, object]:
    """Run one named fixture through the production-wired worker on an isolated ledger.

    The real t6 API (caches, matching, Job Order raise); only Telegram (a recorder) and the
    classifier (a fixed verdict) are stood in, so the message's own ledger, Job Order and
    outbox facts can be checked.
    """
    if missing_secrets():
        return skip(f"production wiring needs every secret; missing {missing_secrets()}"), None
    synthetic = synthetic_update(s.config, text, s.fixture_message_id)
    key = f"{synthetic.chat_id}:{synthetic.message_id}"
    config = s.config.model_copy(update={"dry_run": dry_run})
    telegram = RecordingTelegram([synthetic])
    adapters = build_adapters(config)
    api = ReadOnlyApi(adapters.t6api) if dry_run else adapters.t6api
    adapters = adapters.with_(telegram=telegram, classifier=AlwaysJoClassifier(), t6api=api)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(Path(tmp) / "smoke.db")
        try:
            worker = build_worker(config, ledger, adapters)
            if absent := missing_wiring(worker):
                return fail(f"smoke worker lacks production wiring: {absent}"), telegram
            for _ in range(3):  # poll, drain, deliver; a later cycle only finishes the tail
                await worker.run_once()
                msg = ledger.get_message(key)
                if msg and msg["state"] in ("done", "ignored", "exhausted"):
                    break
                if dry_run and msg and msg["state"] == "planned":
                    break
            result = await verify_capture(
                ledger,
                key,
                job_orders=adapters.t6api,
                expect_dry_run=dry_run,
                recorded_reactions=telegram.reactions,
            )
            msg = ledger.get_message(key)
            s.smoke_page_key = key if msg and msg["notion_page_id"] else None
            if msg and msg["notion_page_id"]:
                page = await adapters.t6api.read_job_order(msg["notion_page_id"])
                print(f"    created {page_url_for(msg['notion_page_id'])} — delete it by hand")
                if "smoke-secret" in json.dumps(page):
                    return fail("credential reached the Job Order"), telegram
        finally:
            await adapters.close()
            ledger.close()
    return result, telegram


async def step_api_access(s: Smoke) -> Result:
    """The API key reads People, Projects and Departments and answers ``jo find``."""
    if missing_secrets():
        return skip(f"needs T6_API_URL and T6_API_SECRET; missing {missing_secrets()}")
    from t6.client import Client

    from jo_watchdog.adapters.t6api import T6Api

    api = T6Api(Client(t6_api_url(), t6_api_secret()))
    counts = {}
    for label, read in (
        ("People", api.people),
        ("Projects", api.projects),
        ("Departments", api.departments),
    ):
        try:
            counts[label] = len(await read())
        except Exception as exc:
            return fail(f"the API key cannot read {label}: {type(exc).__name__}: {exc}")
    try:
        await api.find_job_orders(s.config.intake_chat_id, s.fixture_message_id)
    except Exception as exc:
        return fail(f"jo find failed: {type(exc).__name__}: {exc}")
    return ok(", ".join(f"{n} {label}" for label, n in counts.items()))


async def step_pipeline(s: Smoke) -> Result:
    """Synthetic in-scope message: ledger, plan, Job Order raise, read-back, outbox, reaction."""
    print(f"    fixture message {s.config.intake_chat_id}:{s.fixture_message_id}")
    result, _ = await synthetic_capture(
        s, f"{SMOKE_MARK}\nPurpose: smoke check\npw: smoke-secret", dry_run=False
    )
    if result.status == PASS:
        result.detail += " (Telegram recorded, not live)"
    return result


async def step_dry_run(s: Smoke) -> Result:
    """Dry-run stops at `planned`: one ops post, no Job Order raise, no reaction."""
    result, telegram = await synthetic_capture(s, f"{SMOKE_MARK}\npw: smoke-secret", dry_run=True)
    if result.status != PASS:
        return result
    posts = [text for _, _, text in telegram.sent if DRY_RUN_HEADER in text]
    if telegram.reactions or len(posts) != 1:
        return fail(f"{len(telegram.reactions)} reaction(s), {len(posts)} dry-run post(s)")
    if "smoke-secret" in posts[0]:
        return fail("dry-run post leaked a credential")
    if s.bot and s.args.live_telegram:
        await s.bot.send_message(
            s.config.ops_chat_id, posts[0], message_thread_id=s.config.ops_topic_id,
            parse_mode="HTML",
        )
        return ok("held at planned; plan posted to the ops topic")
    return ok("held at planned; one plan rendered, no Job Order raised, no reaction")


async def step_query_visibility(s: Smoke) -> Result:
    if not s.smoke_page_key:
        return skip("needs the pipeline step's Job Order")
    if missing_secrets():
        return skip(f"needs T6_API_URL and T6_API_SECRET; missing {missing_secrets()}")
    from t6.client import Client

    from jo_watchdog.adapters.t6api import T6Api

    api = T6Api(Client(t6_api_url(), t6_api_secret()))
    chat_id, message_id = (int(part) for part in s.smoke_page_key.split(":"))
    for attempt in range(6):
        if await api.find_job_orders(chat_id, message_id):
            return ok(f"the new Job Order shows in jo find after ~{attempt * 5}s")
        await asyncio.sleep(5)
    return fail("the new Job Order is not found by jo find after 25s")


def _print_check(check: ModelCheck) -> Result:
    for line in check.lines:
        print(f"    {line}")
    return check.result


async def step_extraction(s: Smoke) -> Result:
    """Two live ``llm run`` calls through the production classifier."""
    if not os.environ.get("T6_API_URL") or not os.environ.get("T6_API_SECRET"):
        return skip("T6_API_URL or T6_API_SECRET not set")
    from t6.client import Client

    client = Client(t6_api_url(), t6_api_secret(), timeout=s.config.llm_timeout_seconds)
    return _print_check(
        await check_extraction(
            client,
            model_hint=s.config.llm_model_hint,
            timeout=s.config.llm_timeout_seconds,
        )
    )


async def step_jev(s: Smoke) -> Result:
    """Three live ``llm decide`` calls, one for each regression pair; needs no Notion access."""
    if not os.environ.get("T6_API_URL") or not os.environ.get("T6_API_SECRET"):
        return skip("T6_API_URL or T6_API_SECRET not set")
    from t6.client import Client

    client = Client(t6_api_url(), t6_api_secret(), timeout=s.config.llm_timeout_seconds)
    return _print_check(
        await check_decide(
            client,
            ROOT / "tests/fixtures/jev/regression_pairs.json",
            timeout=s.config.llm_timeout_seconds,
            t_link=s.config.t_link,
            t_hint=s.config.t_hint,
        )
    )


async def step_threshold_tuning(s: Smoke) -> Result:
    """Score the recorded Jev answers offline, retain the results; the owner reviews them."""
    fixtures = ROOT / "tests/fixtures"
    pairs = json.loads((fixtures / "jev/regression_pairs.json").read_text(encoding="utf-8"))
    out = ROOT / "docs/tuning/threshold-results.json"
    report = build_report(
        samples=match_samples(pairs),
        current={"t_link": s.config.t_link, "t_hint": s.config.t_hint},
        sample_source="recorded regression pairs (synthesized until a live capture is retained)",
        previous=read_report(out),
    )
    write_report(out, report)
    results = report["results"]
    print(f"    {results['match_samples']} recorded Jev answers")
    print(f"    results retained at {out.relative_to(ROOT)} (digest {report['results_digest']})")
    review = report["owner_review"]
    if review["status"] != "reviewed":
        return manual(
            "owner must review the link/hint thresholds in the retained results and "
            "attest (actor, time, scope, evidence); linking stays disabled"
        )
    return ok(f"owner review by {review['reviewed_by']} covers digest {report['results_digest']}")


class OnlyMessage:
    """Passes through only the named intake message so a live run cannot capture others."""

    def __init__(self, telegram, chat_id: int, message_id: int):
        self._telegram = telegram
        self._identity = (chat_id, message_id)

    def __getattr__(self, name):
        return getattr(self._telegram, name)

    async def get_updates(self, offset: int, timeout: int):
        updates = await self._telegram.get_updates(offset, timeout)
        return [u for u in updates if (u.chat_id, u.message_id) == self._identity]


async def step_live_telegram(s: Smoke) -> Result:
    if not s.args.live_telegram:
        return skip("pass --live-telegram --live-message-id <id> after posting a test JO")
    if not s.args.live_message_id:
        return skip("--live-message-id is required: the Telegram message id of the post to capture")
    if missing_secrets():
        return skip(f"production wiring needs every secret; missing {missing_secrets()}")
    chat_id, message_id = s.config.intake_chat_id, s.args.live_message_id
    key = f"{chat_id}:{message_id}"
    adapters = build_adapters(s.config)
    adapters = adapters.with_(telegram=OnlyMessage(adapters.telegram, chat_id, message_id))
    with tempfile.TemporaryDirectory() as tmp:
        # Isolated ledger: the service's real ledger and offset are never touched.
        ledger = Ledger(Path(tmp) / "smoke-live.db")
        try:
            worker = build_worker(s.config, ledger, adapters)
            if absent := missing_wiring(worker):
                return fail(f"smoke worker lacks production wiring: {absent}")
            print(f"    waiting up to {s.config.poll_timeout}s for {key} in the intake topic…")
            for _ in range(3):
                await worker.run_once()
                msg = ledger.get_message(key)
                if msg and msg["state"] in ("done", "planned", "ignored", "exhausted"):
                    break
            msg = ledger.get_message(key)
            if msg is None:
                return fail(f"{key} did not arrive; check the id and the intake chat/topic")
            if msg["state"] == "ignored":
                return manual(f"{key} was classified not-a-JO; post a real JO and re-run")
            result = await verify_capture(
                ledger, key, job_orders=adapters.t6api, expect_dry_run=s.config.dry_run
            )
        finally:
            await adapters.close()
            ledger.close()
    if result.status != PASS:
        return result
    await s.bot.send_message(
        s.config.ops_chat_id,
        f"🧪 Smoke test: captured {key}.",
        message_thread_id=s.config.ops_topic_id,
    )
    return ok(f"{result.detail}; ops line posted")


async def _relay_ops_posts(s: Smoke, name: str, posts: list[str]) -> None:
    """With --live-telegram, post the scenario's rendered ops lines to the real ops topic."""
    if not (s.bot and s.args.live_telegram):
        return
    for text in posts:
        await s.bot.send_message(
            s.config.ops_chat_id,
            f"🧪 Smoke scenario {name} (controlled, not a real event)\n{text}",
            message_thread_id=s.config.ops_topic_id,
            parse_mode="HTML",
        )


def scenario_step(name: str) -> Callable[[Smoke], Awaitable[Result]]:
    """A controlled operational scenario (#62): production wiring, scripted failures, no network."""

    async def step(s: Smoke) -> Result:
        run = await SCENARIOS[name](s.config)
        if run.result.status == PASS:
            await _relay_ops_posts(s, name, run.ops_posts)
        return run.result

    return step


def pending(ticket: str, what: str) -> Callable[[Smoke], Awaitable[Result]]:
    async def step(_: Smoke) -> Result:
        return skip(f"{what} — not automated yet ({ticket}); check by hand")

    return step


step_readback_mismatch = scenario_step("readback_mismatch")
step_outbox_retry = scenario_step("outbox_retry")
step_heartbeat_format = scenario_step("heartbeat_format")
step_matcher_failure = scenario_step("matcher_failure")
step_shadow_mode = scenario_step("shadow_mode")
step_dry_run_scenario = scenario_step("dry_run_scenario")

STEPS: list[tuple[str, Callable[[Smoke], Awaitable[Result]]]] = [
    ("config", step_config),
    ("storage", step_storage),
    ("reactions", step_reactions),
    ("reaction_apply", step_reaction_apply),
    ("bot_rights", step_bot_rights),
    ("owner_handle", step_owner_handle),
    ("api_access", step_api_access),
    ("pipeline", step_pipeline),
    ("query_visibility", step_query_visibility),
    ("extraction", step_extraction),
    ("jev", step_jev),
    ("dry_run", step_dry_run),
    ("dry_run_scenario", step_dry_run_scenario),
    ("live_telegram", step_live_telegram),
    ("threshold_tuning", step_threshold_tuning),
    ("readback_mismatch", step_readback_mismatch),
    ("outbox_retry", step_outbox_retry),
    ("forward_origins", step_forward_origins),
    ("channel_link", step_channel_link),
    ("heartbeat_format", step_heartbeat_format),
    ("matcher_failure", step_matcher_failure),
    ("shadow_mode", step_shadow_mode),
]


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    only = set(args.only.split(",")) if args.only else None
    names = [name for name, _ in STEPS]
    unknown = (only or set()) - set(names)
    if unknown:
        print(f"unknown steps: {', '.join(sorted(unknown))}")
        return 2
    try:
        attestations = (
            load_attestations(args.attestations, set(names)) if args.attestations else {}
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"bad attestations file: {exc}")
        return 2
    s = Smoke(config, args)
    s.attestations = attestations
    await s.open()
    results: dict[str, Result] = {}
    try:
        for name, step in STEPS:
            if only and name not in only:
                continue
            try:
                result = await step(s)
            except Exception as exc:  # one broken step must not hide the rest
                result = fail(f"{type(exc).__name__}: {exc}")
            result = attest(result, attestations.get(name))
            results[name] = result
            print(f"{result.status:<8} {name}: {result.detail}")
    finally:
        await s.close()
    counts: dict[str, int] = {}
    for result in results.values():
        counts[result.status] = counts.get(result.status, 0) + 1
    print("\n" + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    final = verdict(results, names, selected_only=bool(only))
    print(f"\n{final.label} VERIFICATION" + "".join(f"\n  {r}" for r in final.reasons))
    return final.exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--only", help="comma-separated step names to run")
    parser.add_argument("--live-telegram", action="store_true", help="capture one real post")
    parser.add_argument(
        "--live-message-id",
        type=int,
        help="with --live-telegram: the Telegram message id of the one post to capture",
    )
    parser.add_argument(
        "--fixture-message-id",
        type=int,
        help="message id for the synthetic fixture (default: derived from the clock)",
    )
    parser.add_argument(
        "--forward-fixtures",
        help="JSON {user, hidden_user, chat, channel: raw Bot API message of a real forward}; "
        "an omitted origin is a gap, not a pass",
    )
    parser.add_argument(
        "--reaction-message-id",
        type=int,
        help="a real intake-group message to apply the configured reaction to",
    )
    parser.add_argument(
        "--reaction",
        choices=("captured", "flagged"),
        default="captured",
        help="which configured reaction reaction_apply sets (default: captured)",
    )
    parser.add_argument(
        "--attestations",
        help="JSON list of {step, actor, at, scope, evidence} settling SKIP/MANUAL steps by hand",
    )
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
