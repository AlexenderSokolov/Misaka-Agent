"""Where each possibility went, and who may say so (schema v20).

A possibility -- an option, a divergence proposal, a red-team issue taken to a decision -- points at
most once at what answers it, and following the pointers ends at a node or at a record left
unopened. A node judges coverage on its own line, whose deliberation it holds; a claim that a node
of another line covers a proposal goes to the reconciliation, the one place both lines are in view,
which upholds it or reopens the proposal as a possibility of its own. An option left unopened names
the node that answers it, and an option that brings back a declined path names it. Structure only:
nothing here judges whether two possibilities are the same.
"""
from contextlib import closing
from types import SimpleNamespace

import pytest

from misaka.core import graph
from misaka.core.platform import tasks
from misaka.core.research import commands, context, planner, runs, views, workflow


@pytest.fixture
def board(tmp_path):
    with closing(tasks.connect(str(tmp_path / "board.db"))) as con:
        yield con


def _close(con, node):
    con.execute("UPDATE research_execution SET status='closed' WHERE id=?", (node["id"],))
    return runs.node(con, node["id"])


def _conclude(con, run, node, name="synthesis.md"):
    runs.write_text(con, run["id"], "synthesis", "Conclusion", runs.generated_path(name, node_id=node["id"]),
                    f"Conclusion of {node['id']}", branch_id=node["id"])


def _proposal(con, run, node, question):
    return runs.add_issue(con, run["id"], node=node, task_id=f"div-{node['id']}", round=1, origin="divergence",
                          kind="frame", question=question, rationale=f"Premise: {question}")


def _decide(con, run, node, payload, tmp_path):
    """A node's decision as the workflow records and applies it, then the node closes."""
    payload = commands.Decide.model_validate(payload).model_dump()
    runs.record_action(con, run, node, runs.DECIDE_KEY, payload, session_file=str(tmp_path / f"{node['id']}.jsonl"),
                       tool_call_id=f"decide-{node['id']}")
    workflow._apply_decisions(con, run, node, node, payload)
    return _close(con, node)


def _fork(question, *labels):
    return {"question": question, "stakes": "s",
            "options": [{"label": "as concluded", "premise": "own", "own": True},
                        *({"label": label, "premise": f"premise of {label}"} for label in labels)]}


def _reconcile(con, run, round, payload, tmp_path):
    payload = {"nodes": [], "joins": [], "relations": [], "not_pursued": [], "coverage": [], "summary": "s", **payload}
    resolved = views.resolve_reconcile(con, run, payload)
    runs.record_reconcile(con, run, round, payload, session_file=str(tmp_path / "root.jsonl"),
                          tool_call_id=f"reconcile-{round}")
    _needed, considered = views.reconcile_state(con, run)
    return resolved, runs.apply_reconcile(con, run, round, resolved, considered=considered)


@pytest.fixture
def level(board, tmp_path):
    """The root forked into A and B, both opened by the first reconciliation: two lines, one level."""
    run = runs.create(board, workspace=str(tmp_path), question="Why did the reform fail?", limits={"max_depth": 3})
    assert runs.acquire_driver(board, run["id"], "driver:test")
    run, root = runs.get(board, run["id"]), runs.root(board, run["id"])
    _conclude(board, run, root)
    root = _decide(board, run, root, {"decisions": [_fork("Which cause?", "fiscal", "coalition")]}, tmp_path)
    _own, fiscal, coalition = (row["id"] for row in runs.options(board, run["id"]))
    _resolved, receipt = _reconcile(board, run, 1, {"nodes": [
        {"question": "Fiscal?", "options": [fiscal], "rationale": "r"},
        {"question": "Coalition?", "options": [coalition], "rationale": "r"}]}, tmp_path)
    a, b = (runs.node(board, node_id) for node_id in receipt["nodes"])
    _conclude(board, run, a)
    _conclude(board, run, b)
    return SimpleNamespace(con=board, run=runs.get(board, run["id"]), root=root, a=a, b=b, tmp=tmp_path)


def _paths(con, run):
    return views.render_paths(views.snapshot(con, runs.get(con, run["id"])))


def test_coverage_by_another_line_is_a_claim_and_by_the_own_line_is_not(level):
    con, run, root, a, b = level.con, level.run, level.root, level.a, level.b
    across, own_line = _proposal(con, run, a, "Patronage networks"), _proposal(con, run, a, "Fiscal capacity")
    _decide(con, run, a, {"declined": [
        {"issue_id": across, "disposition": "covered", "covered_by": b["id"], "reason": "B studies coalitions"},
        {"issue_id": own_line, "disposition": "covered", "covered_by": root["id"], "reason": "the root's line"}]},
        level.tmp)
    _decide(con, run, b, {"decisions": []}, level.tmp)
    [claim] = views.coverage_claims(con, run)
    assert (claim["proposal"], claim["raised_at"], claim["covered_by"]["node"]) == (across, a["id"], b["id"])
    assert claim["covered_by"]["conclusion"].endswith("synthesis.md") and not claim["covered_by"]["revised_since_claim"]
    # Nothing else calls for the root: no option waits and the level's conclusions were reconciled... but the claim does.
    assert views.reconcile_state(con, run)[0] is True
    material = workflow._reconcile_material(con, run)
    assert [item["proposal"] for item in material["coverage_claims"]] == [across]
    assert "a claim across lines: the next reconciliation reviews it" in _paths(con, run)


def test_a_claim_records_the_covering_conclusion_written_after_it(level):
    con, run, a, b = level.con, level.run, level.a, level.b
    issue = _proposal(con, run, a, "Patronage networks")
    _decide(con, run, a, {"declined": [
        {"issue_id": issue, "disposition": "covered", "covered_by": b["id"], "reason": "B studies it"}]}, level.tmp)
    _conclude(con, run, b, "synthesis-2.md")       # B revises after A decided
    con.execute("UPDATE research_artifacts SET created_at=created_at+100 WHERE path LIKE '%synthesis-2.md'")
    [claim] = views.coverage_claims(con, run)
    assert claim["covered_by"]["revised_since_claim"] is True


def test_every_claim_gets_one_verdict_and_a_reopened_one_an_outcome(level):
    con, run, a, b = level.con, level.run, level.a, level.b
    held, reopened = _proposal(con, run, a, "Patronage networks"), _proposal(con, run, a, "Moral economy")
    _decide(con, run, a, {"declined": [
        {"issue_id": held, "disposition": "covered", "covered_by": b["id"], "reason": "B studies patronage"},
        {"issue_id": reopened, "disposition": "covered", "covered_by": b["id"], "reason": "B studies coalitions"}]},
        level.tmp)
    _decide(con, run, b, {"decisions": []}, level.tmp)
    verdicts = [{"proposal": held, "verdict": "holds", "note": "B's section 2 reads the networks"},
                {"proposal": reopened, "verdict": "reopen", "note": "it reads the revolt from the villagers' norms"}]

    def check(**payload):
        return views.resolve_reconcile(con, run, {"nodes": [], "joins": [], "not_pursued": [], "relations": [],
                                                  "coverage": [], **payload})

    with pytest.raises(ValueError, match="every coverage claim needs a verdict"):
        check(coverage=verdicts[:1])
    with pytest.raises(ValueError, match="more than one verdict"):
        check(coverage=[*verdicts, verdicts[0]])
    with pytest.raises(ValueError, match="not coverage claims"):
        check(coverage=[*verdicts, {"proposal": "i_nope", "verdict": "holds", "note": "?"}])
    with pytest.raises(ValueError, match="every pending option, and every proposal whose coverage is reopened"):
        check(coverage=verdicts)
    with pytest.raises(ValueError, match="or proposals whose coverage this reconciliation reopens"):
        check(coverage=verdicts, nodes=[{"question": "Q", "options": [reopened, held], "rationale": "r"}])

    resolved, receipt = _reconcile(con, run, 2, {"coverage": verdicts, "nodes": [
        {"question": "What norms did the villagers hold?", "options": [reopened], "rationale": "not B's line"}]},
        level.tmp)
    assert resolved["nodes"][0]["parents"] == [a["id"]]
    child = runs.node(con, receipt["nodes"][0])
    assert child["depth"] == 2 and [p["id"] for p in runs.parents(con, child["id"])] == [a["id"]]
    # The reopened proposal is now an option of a decision the reconciliation recorded at the node that raised it.
    [option] = runs.origin_options(con, child["id"])
    assert (option["decided_at"], option["decision_origin"], option["premise"]) == (
        a["id"], "reconcile", "it reads the revolt from the villagers' norms")
    issue = runs.issue(con, reopened)
    assert (issue["disposition"], issue["option_id"], issue["covered_by"]) == ("branch", option["id"], None)
    assert issue["coverage_review"]["verdict"] == "reopen" and issue["coverage_review"]["claimed"] == b["id"]
    assert receipt["reopened"] == {reopened: option["id"]}
    assert f"takes up `{reopened}` (raised at node {a['id']})" in planner.position(con, run, child)
    # The claim that held keeps its pointer and says so; neither claim is asked about again.
    assert runs.issue(con, held)["covered_by"] == b["id"]
    assert views.coverage_claims(con, run) == []
    assert "the coverage claim held at reconciliation 2: B's section 2 reads the networks" in _paths(con, run)


def test_a_reopened_claim_may_stay_unopened_for_a_structural_reason(level):
    con, run, a, b = level.con, level.run, level.a, level.b
    issue = _proposal(con, run, a, "Moral economy")
    _decide(con, run, a, {"declined": [
        {"issue_id": issue, "disposition": "covered", "covered_by": b["id"], "reason": "B"}]}, level.tmp)
    _decide(con, run, b, {"decisions": []}, level.tmp)
    _reconcile(con, run, 2, {"coverage": [{"proposal": issue, "verdict": "reopen", "note": "villagers' norms"}],
                             "not_pursued": [{"option": issue, "reason": "the node limit the user set"}]}, level.tmp)
    option = runs.option(con, runs.issue(con, issue)["option_id"])
    assert option["reason"] == "the node limit the user set" and option["node_id"] is None


def test_a_claim_on_another_node_s_waiting_option_is_reviewed_too(level):
    con, run, a, b = level.con, level.run, level.a, level.b
    a = _decide(con, run, a, {"decisions": [_fork("Which revenue?", "salt tax")]}, level.tmp)
    salt = next(row["id"] for row in runs.pending_options(con, run["id"]))
    issue = _proposal(con, run, b, "Salt monopoly")
    _decide(con, run, b, {"declined": [
        {"issue_id": issue, "disposition": "covered", "covered_by": salt, "reason": "A's salt tax option"}]}, level.tmp)
    [claim] = views.coverage_claims(con, run)
    assert claim["covered_by"]["option"] == salt and claim["covered_by"]["decided_at"] == a["id"]
    _resolved, receipt = _reconcile(con, run, 2, {
        "coverage": [{"proposal": issue, "verdict": "holds", "note": "the same monopoly"}],
        "nodes": [{"question": "Salt?", "options": [salt], "rationale": "r"}]}, level.tmp)
    destinations = views.snapshot(con, runs.get(con, run["id"]))["destinations"]
    [group] = [g for g in destinations["nodes"] if g["node"] == receipt["nodes"][0]]
    [via] = [m for m in group["members"] if m["id"] == issue]
    assert via["via"] == [salt]                      # the proposal reaches the node through the option


def test_an_option_left_unopened_names_what_answers_it(level):
    con, run, a, b = level.con, level.run, level.a, level.b
    a = _decide(con, run, a, {"decisions": [_fork("Which revenue?", "salt tax", "customs")]}, level.tmp)
    b = _decide(con, run, b, {"decisions": [_fork("Whose coalition?", "gentry")]}, level.tmp)
    salt, customs, gentry = (row["id"] for row in runs.pending_options(con, run["id"]))

    def outcome(**item):
        return {"option": customs, "reason": "complements B", **item}

    nodes = [{"question": "Salt?", "options": [salt], "rationale": "r"}]
    base = {"nodes": nodes, "joins": [], "relations": [], "coverage": [],
            "not_pursued": [{"option": gentry, "reason": "the node limit"}]}
    with pytest.raises(ValueError, match="neither a node of this run nor an option this reconciliation opens"):
        views.resolve_reconcile(con, run, {**base, "not_pursued": [*base["not_pursued"], outcome(covered_by=gentry)]})
    with pytest.raises(ValueError, match="covered by itself"):
        views.resolve_reconcile(con, run, {**base, "not_pursued": [*base["not_pursued"], outcome(covered_by=customs)]})
    _reconcile(con, run, 2, {"nodes": nodes, "not_pursued": [outcome(covered_by=b["id"]),
                                                             {"option": gentry, "reason": "same as salt",
                                                              "covered_by": salt}]}, level.tmp)
    assert runs.option(con, customs)["covered_by"] == b["id"] and runs.option(con, gentry)["covered_by"] == salt
    paths = _paths(con, run)
    assert f"not pursued, answered by node `{b['id']}`: complements B" in paths
    assert f"not pursued, answered by option `{salt}`: same as salt" in paths
    # Followed to its end, the option left unopened stands with the node that answers it.
    data = views.snapshot(con, runs.get(con, run["id"]))
    under_b = next(g for g in data["destinations"]["nodes"] if g["node"] == b["id"])
    assert customs in [m["id"] for m in under_b["members"]]
    salt_node = runs.option(con, salt)["node_id"]
    under_salt = next(g for g in data["destinations"]["nodes"] if g["node"] == salt_node)
    assert {m["id"]: m["via"] for m in under_salt["members"]} == {salt: [], gentry: [salt]}
    assert "## Where each possibility went" in paths and f"### Node `{salt_node}`" in paths
    packet = context.render(context.build(con, run, node=runs.node(con, salt_node)))
    assert f"not pursued, answered by {b['id']}: complements B" in packet


def test_a_path_declined_comes_back_only_naming_what_it_answers(level, monkeypatch):
    con, run, a, b = level.con, level.run, level.a, level.b
    declined = _proposal(con, run, a, "Archive of petitions")
    _decide(con, run, a, {"declined": [{"issue_id": declined, "disposition": "decline",
                                        "reason": "the petitions are not digitised"}]}, level.tmp)
    again = _proposal(con, run, b, "Petitions, read in the printed edition")
    captured = {}

    def command(con_, run_, cfg, worker_, node, prompt, *, validate, **_kwargs):
        captured.update(validate=validate, prompt=prompt)
        raise RuntimeError("captured")

    monkeypatch.setattr(planner, "_command", command)
    with pytest.raises(RuntimeError, match="captured"):
        planner.decide(con, run, {}, None, b, divergence=[], proposals=[runs.issue(con, again)], branches=[])
    validate = captured["validate"]

    def option(**extra):
        return {"decisions": [{"question": "Which archive?", "stakes": "s", "options": [
            {"label": "as concluded", "premise": "own", "own": True, **extra.pop("own_extra", {})},
            {"label": "printed petitions", "premise": "the 1911 edition answers the digitisation", "sources": [again],
             **extra}]}]}

    def check(payload):
        return validate(commands.Decide.model_validate(payload).model_dump())

    with pytest.raises(ValueError, match="does not show declined, covered or left unopened"):
        check(option(answers=["i_nope"]))
    with pytest.raises(ValueError, match="brings back no path"):
        check(option(own_extra={"answers": [declined]}))
    payload = check(option(answers=[declined]))
    _decide(con, run, b, payload, level.tmp)
    brought = next(row for row in runs.pending_options(con, run["id"]))
    assert brought["answers"] == [declined]
    material = workflow._reconcile_material(con, run)
    [pending] = material["pending_options"]
    assert pending["answers"][0]["path"] == declined and "not digitised" in pending["answers"][0]["went"]
    paths = _paths(con, run)
    assert f"brought back by option `{brought['id']}` at `{b['id']}`, which answers the reason recorded here" in paths
    assert declined in views.reopenable(con, run) and brought["id"] not in views.reopenable(con, run)


def test_the_decision_knows_its_own_line_and_where_each_judgement_was_made(level, monkeypatch):
    """A node holds its own line's deliberation in the conversation it continues -- a join's parents' only in
    their sessions. The decision is told which, so it reads why a path went where it went before calling a
    proposal the same, and the prompt names no conversation tool (LCM is pluggable)."""
    con, run, root, a, b = level.con, level.run, level.root, level.a, level.b
    a = _decide(con, run, a, {"decisions": [_fork("Which revenue?", "salt tax")]}, level.tmp)
    b = _decide(con, run, b, {"decisions": []}, level.tmp)
    _salt = next(row["id"] for row in runs.pending_options(con, run["id"]))
    _resolved, receipt = _reconcile(con, run, 2, {
        "not_pursued": [{"option": _salt, "reason": "the node limit"}],
        "joins": [{"parents": [a["id"], b["id"]], "question": "Fiscal coalition?", "rationale": "they meet"}]},
        level.tmp)
    join = runs.node(con, receipt["joins"][0])
    assert views.lineage(con, join) == [join["id"], root["id"]]
    assert views.lineage(con, a) == [a["id"], root["id"]]
    line = {entry["id"]: entry for entry in planner.on_your_line(con, run, join)}
    from_root = [e for e in line.values() if e["decided_at"] == root["id"]]
    from_a = [e for e in line.values() if e["decided_at"] == a["id"]]
    assert from_root and all(e["deliberation"]["in"] == "this conversation, above" for e in from_root)
    assert from_a and all(e["deliberation"]["in"] == f"node {a['id']}'s conversation" for e in from_a)
    assert all(e["deliberation"]["session_file"].endswith(f"{a['id']}.jsonl") for e in from_a)
    assert line[_salt]["went"] == "not pursued: the node limit"
    captured = {}

    def command(con_, run_, cfg, worker_, node, prompt, *, validate, **_kwargs):
        captured["prompt"] = prompt
        raise RuntimeError("captured")

    monkeypatch.setattr(planner, "_command", command)
    with pytest.raises(RuntimeError, match="captured"):
        planner.decide(con, run, {}, None, join, divergence=[], proposals=[], branches=[])
    assert "on_your_line" in captured["prompt"] and "this conversation, above" in captured["prompt"]
    for contract in (planner.DECIDE_CONTRACT, planner.RECONCILE_CONTRACT, planner.DIVERGENCE_CONTRACT):
        assert "lcm_" not in contract


def test_a_red_team_issue_declined_at_the_decision_stays_on_the_paths_list(level):
    """Before v20 the list kept a red-team issue only while it was disposed `branch`: declined or covered at
    the decision, it fell off the list it is meant to keep from being raised again."""
    con, run, a = level.con, level.run, level.a
    issue = runs.add_issue(con, run["id"], node=a, task_id="red", round=1, origin="critique", kind="frame",
                           question="The framework breaks", rationale="r")
    runs.dispose(con, issue, "branch", reason="another frame")
    _decide(con, run, a, {"declined": [{"issue_id": issue, "disposition": "decline", "reason": "out of scope"}]},
            level.tmp)
    assert issue in [row["id"] for row in views.possibility_issues(con, run)]
    assert f"`{issue}` proposal: The framework breaks → declined: another frame -- not opened: out of scope" in \
        _paths(con, run)


def test_contracts_route_coverage_to_whoever_can_see_both_lines():
    flat = lambda text: " ".join(text.split())
    decide, reconcile = flat(planner.DECIDE_CONTRACT), flat(planner.RECONCILE_CONTRACT)
    assert "Your own line you know from the inside" in decide and "`on_your_line`" in decide
    assert "is a claim the reconciliation reviews with both lines in view" in decide
    assert "name it in the option's `answers`" in decide and "Being covered is not being answered" in decide
    assert "Give every claim below a verdict in `coverage`" in reconcile
    assert "that node's id in `covered_by`" in reconcile
    assert "each coverage claim it upholds or reopens" in flat(planner.RECONCILE_WAITS)
    assert "groups every possibility under what answers it now" in flat(planner.DIVERGENCE_CONTRACT)


def test_a_v19_board_gets_the_v20_graph_schema(tmp_path):
    board = str(tmp_path / "board.db")
    with closing(tasks.connect(board)) as con:
        run = runs.create(con, workspace=str(tmp_path), question="Why?", limits={"max_depth": 2})
        old = runs.graph_schema(runs.limits(run))
        old["edges"]["covered_by"] = {"from": ["issue"], "to": ["inquiry"], "max_out": 1}
        del old["edges"]["answers"]
        with runs._holding(con, run["id"]) as token:
            graph.commit(con, run["id"], "schema", old, token=token)
        assert "answers" not in graph.get(con, run["id"]).schema.edges
        con.execute("DELETE FROM schema_migrations WHERE component='research'")
        con.execute("INSERT INTO schema_migrations(component,version,applied_at) VALUES('research',19,0)")
    with closing(tasks.connect(board)) as con:
        runs.init(con)
        schema = graph.get(con, run["id"]).schema
        assert "answers" in schema.edges and schema.edges["covered_by"].source == {"issue", "option"}
        assert con.execute("SELECT MAX(version) FROM schema_migrations WHERE component='research'").fetchone()[0] == 20
        root = runs.root(con, run["id"])
        did = runs.add_decision(con, run["id"], node=root, round=1, origin="decide", index=0, question="Q",
                                stakes="s", options=[{"label": "x", "premise": "p"}, {"label": "y", "premise": "q"}])
        x, _y = (row["id"] for row in runs.options(con, run["id"], decision_id=did))
        runs.set_option(con, x, reason="covered", covered_by=root["id"])
        assert runs.option(con, x)["covered_by"] == root["id"]
