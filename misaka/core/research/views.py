"""The research graph: queries over its nodes and edges, the structural check of a level's
reconciliation, and the views written into the project.

SQLite (``runs``) is the only authority. ``final/<run>-graph.md``, ``nodes/<node>/NODE.md`` and
``final/<run>-graph.json`` are rewritten from it after state changes and never read back, like the
source bundles. Python checks structure here -- references, depth, the node budget, that every
pending option gets an outcome -- and never judges what a node found or whether two nodes ask the
same question: that is Last Order's call.
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from contextlib import contextmanager
from itertools import pairwise

from misaka.config import home
from misaka.core.research import runs
from misaka.utils import atomic, file_lock
from misaka.utils.paths import posix_relpath

_LOG = logging.getLogger(__name__)

GRAPH_VIEW = "graph.md"
GRAPH_SNAPSHOT = "graph.json"
PATHS_VIEW = "paths.md"
NODE_VIEW = "NODE.md"


def kind(con, node):
    """``root`` (depth 0), ``join`` (several parents) or ``alt`` (one parent): read off the edges."""
    if runs.is_root(node):
        return "root"
    return "join" if len(runs.parents(con, node["id"])) > 1 else "alt"


def ancestors(con, node_id):
    """Every node the given one descends from, through any parent."""
    seen, frontier = set(), [node_id]
    while frontier:
        for parent in runs.parents(con, frontier.pop()):
            if parent["id"] not in seen:
                seen.add(parent["id"])
                frontier.append(parent["id"])
    return seen


def fork_source(con, node):
    """The node whose Last Order conversation a new node forks: the lowest common ancestor of its
    parents, which for a single parent is that parent. Parents can have several lowest common
    ancestors, none descending from another; each carries its own line only, so the search goes on
    to the lowest node they all descend from. A join thus inherits what its lines share and neither
    line's own deliberation, which reaches it through its context packet. The root ends every search."""
    ids = [parent["id"] for parent in runs.parents(con, node["id"])]
    while len(ids) > 1:
        common = set.intersection(*({i} | ancestors(con, i) for i in ids))
        above = {c: ancestors(con, c) for c in common}
        ids = sorted(c for c in common if not any(c in above[other] for other in common if other != c))
    return runs.node(con, ids[0])


def syntheses(con, run, node):
    """The node's conclusions, first version first."""
    return list(runs.artifacts(con, run["id"], branch_id=node["id"], kind="synthesis"))


def dissolution(con, run, node):
    """What the node's latest conclusion dissolves, when it does: ``{"question", "grounds"}`` with
    ``question`` either ``this_node`` or ``research_question``. A revised conclusion that no longer
    dissolves anything leaves the earlier version's record behind."""
    versions = len(syntheses(con, run, node))
    recorded = runs.action(con, run["id"], node["id"], runs.dissolve_key(versions)) if versions else None
    return recorded["payload"] if recorded else None


def dissolving_nodes(con, run):
    """Nodes whose latest conclusion dissolves the research question itself."""
    return [node for node in runs.nodes(con, run["id"])
            if (dissolution(con, run, node) or {}).get("question") == "research_question"]


def graph_path(run):
    return runs.run_path(run, GRAPH_VIEW)


def paths_path(run):
    return runs.run_path(run, PATHS_VIEW)


def node_view_path(node_id):
    return os.path.join(runs.node_dir(node_id), NODE_VIEW)


# --- reconciliation ---------------------------------------------------------------------

def reconcilable_options(con, run):
    """Pending options whose deciding node has finished its own work: what a reconciliation must
    give an outcome to."""
    return [row for row in runs.pending_options(con, run["id"])
            if runs.node(con, row["decided_at"])["status"] == "closed"]


def reconcile_state(con, run):
    """``(needed, considered)``: whether this level barrier needs the root Last Order, and the
    finished nodes a reconciliation now would cover. A barrier: never while a node is still running
    or failed and waiting for its retry -- not even on a resume, which restarts the driver's loop at
    this check with a level half done. Then it is needed while an option waits for an outcome, or
    when nodes finished since the last one and at least two nodes hold a conclusion (below that no
    join or relation can exist). Structure only; nothing here reads what a node found."""
    nodes = runs.nodes(con, run["id"])
    finished = [n for n in nodes if n["status"] == "closed"]
    covered = set()
    for item in runs.reconcile_rounds(con, run["id"]):
        if item["applied_at"] is not None:
            covered.update(item["receipt"]["considered"])
    fresh = [n["id"] for n in finished if n["id"] not in covered]
    concluded = sum(1 for n in finished if syntheses(con, run, n))
    settled = all(n["status"] in ("closed", "parked") for n in nodes)
    needed = settled and (bool(reconcilable_options(con, run)) or (bool(fresh) and concluded >= 2))
    return needed, [n["id"] for n in finished]


def resolve_reconcile(con, run, payload):
    """Check a reconciliation's structure against the graph as it stands and return what applying
    it creates: each new node with its parents (the nodes whose options it pursues, or the nodes
    it joins). Raises ValueError with what to change; judges nothing about content."""
    chosen = runs.current_limits(con, run)
    pending = {row["id"]: row for row in reconcilable_options(con, run)}
    named = Counter([oid for item in payload["nodes"] for oid in item["options"]]
                    + [item["option"] for item in payload["not_pursued"]])
    problems = []
    unknown = sorted(set(named) - set(pending))
    if unknown:
        problems.append(f"not pending options of this run: {', '.join(unknown)}")
    twice = sorted(oid for oid, count in named.items() if count > 1)
    if twice:
        problems.append(f"options given more than one outcome: {', '.join(twice)}")
    missing = sorted(set(pending) - set(named))
    if missing:
        problems.append("every pending option needs an outcome (a node, or not_pursued with its reason); "
                        f"missing: {', '.join(missing)}")
    resolved = {"nodes": [], "joins": [], "not_pursued": [], "relations": []}
    total = len(runs.nodes(con, run["id"]))
    for position, item in enumerate(payload["nodes"], 1):
        if not item["options"]:
            problems.append(f"nodes[{position}] pursues no option; a node opened without one is a join")
            continue
        parent_ids = list(dict.fromkeys(pending[oid]["decided_at"] for oid in item["options"] if oid in pending))
        depth = max((runs.node(con, pid)["depth"] for pid in parent_ids), default=-1) + 1
        if parent_ids and depth > chosen["max_depth"]:
            problems.append(f"nodes[{position}] would open at depth {depth}, beyond max_depth {chosen['max_depth']}: "
                            "record its options under not_pursued with the depth limit as the reason")
        resolved["nodes"].append({"question": item["question"], "parents": parent_ids,
                                  "options": list(item["options"]), "rationale": item["rationale"]})
    for position, item in enumerate(payload["joins"], 1):
        parent_ids = list(dict.fromkeys(item["parents"]))
        rows = [runs.node(con, pid) for pid in parent_ids]
        if len(parent_ids) < 2:
            problems.append(f"joins[{position}] names fewer than two distinct nodes")
        if any(row is None or row["run_id"] != run["id"] for row in rows):
            problems.append(f"joins[{position}] names a node that is not in this run")
            continue
        unfinished = [row["id"] for row in rows if row["status"] != "closed"]
        if unfinished:
            problems.append(f"joins[{position}]: {', '.join(unfinished)} has not finished its own work "
                            "(or was skipped or failed) and cannot carry a join")
        related = [f"{a} descends from {b}" for a in parent_ids for b in parent_ids
                   if a != b and b in ancestors(con, a)]
        if related:
            problems.append(f"joins[{position}]: {'; '.join(related)} -- a node already carries its ancestors "
                            "forward, so join nodes that are not each other's ancestors")
        depth = max((row["depth"] for row in rows), default=-1) + 1
        if depth > chosen["max_depth"]:
            problems.append(f"joins[{position}] would open at depth {depth}, beyond max_depth {chosen['max_depth']}")
        resolved["joins"].append({"question": item["question"], "parents": parent_ids, "rationale": item["rationale"]})
    opened = len(resolved["nodes"]) + len(resolved["joins"])
    if total + opened > chosen["max_nodes"]:
        problems.append(f"this would bring the graph to {total + opened} nodes, over max_nodes {chosen['max_nodes']}: "
                        "merge options that ask the same question into one node, or record options under "
                        "not_pursued with the node limit as the reason")
    for item in payload["not_pursued"]:
        resolved["not_pursued"].append({"option": item["option"], "reason": item["reason"]})
    resolved["relations"] = _relations(con, run, payload["relations"], problems)
    if problems:
        raise ValueError("Reconciliation not accepted: " + "; ".join(problems) + ".")
    return resolved


def _relations(con, run, items, problems):
    out = []
    for position, item in enumerate(items, 1):
        members = list(dict.fromkeys(item["nodes"]))
        rows = [runs.node(con, nid) for nid in members]
        if len(members) < 2 or any(row is None or row["run_id"] != run["id"] for row in rows):
            problems.append(f"relations[{position}] must name at least two distinct nodes of this run")
        out.append({"kind": item["kind"], "nodes": members, "note": item["note"]})
    return out


def resolve_relations(con, run, payload):
    """Check relations recorded outside a reconciliation: every one names two or more nodes of the run."""
    problems = []
    relations = _relations(con, run, payload["relations"], problems)
    if problems:
        raise ValueError("Relations not accepted: " + "; ".join(problems) + ".")
    return {"relations": relations}


# --- views --------------------------------------------------------------------------------

def _clip(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value):
    return " ".join(str(value or "").splitlines()).strip().replace("|", r"\|")


def _mermaid_label(text):
    return _clip(text, 60).replace('"', "'").replace("[", "(").replace("]", ")")


def _outcome(con, option, deciding_node_id):
    if option["node_id"] == deciding_node_id:
        return "this node's own line"
    if option["node_id"]:
        return f"node `{option['node_id']}`"
    if option["reason"]:
        return f"not pursued: {option['reason']}"
    return "pending"


def _dispositions(con, run, node):
    counts = Counter(row["disposition"] or "undisposed" for row in runs.issues(con, run["id"], node_id=node["id"]))
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "-"


def snapshot(con, run):
    """The whole graph as plain data, read in one pass: the source of every view."""
    workspace = run["workspace"]
    rel = lambda path: posix_relpath(path, workspace) if path else None
    out = {"run": run["id"], "question": run["question"], "status": run["status"], "phase": run["phase"],
           "limits": runs.limits(run), "nodes": [], "decisions": [], "relations": runs.relations(con, run["id"]),
           "reconciles": []}
    for node in runs.nodes(con, run["id"]):
        files = {}
        for row in runs.artifacts(con, run["id"], branch_id=node["id"]):
            if row["task_id"] is None:
                files.setdefault(row["kind"], []).append(rel(row["path"]))
        out["nodes"].append({
            "id": node["id"], "question": node["question"], "depth": node["depth"], "status": node["status"],
            "kind": kind(con, node), "parents": [p["id"] for p in runs.parents(con, node["id"])],
            "children": [c["id"] for c in runs.children(con, node["id"])],
            "reached_by": [{"option": o["id"], "label": o["label"], "premise": o["premise"],
                            "decision": o["decision_question"], "decided_at": o["decided_at"]}
                           for o in runs.origin_options(con, node["id"])],
            "files": files, "issues": _dispositions(con, run, node), "dissolves": dissolution(con, run, node),
            "red_team": runs.red_team(con, run["id"], node["id"]),
            "cards": [{"id": t["id"], "title": t["title"], "kind": t["research_kind"], "round": t["round"],
                       "attempt": int(t["generation"]), "status": t["status"], "folder": rel(t["output_dir"])}
                      for t in runs.tasks(con, run["id"], node_id=node["id"])],
        })
    for decision in runs.decisions(con, run["id"]):
        out["decisions"].append({
            "id": decision["id"], "node": decision["node_id"], "origin": decision["origin"],
            "question": decision["question"], "stakes": decision["stakes"],
            "options": [{"id": o["id"], "label": o["label"], "premise": o["premise"], "node": o["node_id"],
                         "reason": o["reason"], "outcome": _outcome(con, o, decision["node_id"])}
                        for o in runs.options(con, run["id"], decision_id=decision["id"])]})
    for item in runs.reconcile_rounds(con, run["id"]):
        out["reconciles"].append({"round": item["round"], "applied": item["applied_at"] is not None,
                                  "summary": item["payload"].get("summary", ""), "receipt": item["receipt"]})
    out["paths"] = _paths(con, run, out)
    out["errata"] = runs.errata(con, run["id"])
    return out


def _paths(con, run, data):
    """Every possibility raised in the run, under the node that raised it, with where it went: each
    option of every decision (with the proposals it takes up), and each divergence proposal or branch
    issue no option took up. Red-team corrections are not possibilities and stay out."""
    options = {option["id"] for decision in data["decisions"] for option in decision["options"]}
    issues = runs.issues(con, run["id"])
    taken = {}
    for issue in issues:
        if issue["option_id"] in options:
            taken.setdefault(issue["option_id"], []).append({"issue": issue["id"], "question": issue["question"]})
    out = []
    for node in data["nodes"]:
        entries = [{"id": option["id"], "what": "option", "text": option["label"], "premise": option["premise"],
                    "decision": decision["question"], "outcome": option["outcome"], "takes_up": taken.get(option["id"], [])}
                   for decision in data["decisions"] if decision["node"] == node["id"] for option in decision["options"]]
        entries += [{"id": issue["id"], "what": "proposal", "text": issue["question"], "outcome": _proposal_outcome(con, run, issue)}
                    for issue in issues if issue["branch_id"] == node["id"]
                    and (issue["origin"] == "divergence" or issue["disposition"] == "branch")
                    and issue["option_id"] not in options]
        if entries:
            out.append({"node": node["id"], "kind": node["kind"], "depth": node["depth"], "question": node["question"],
                        "entries": entries})
    return out


def _proposal_outcome(con, run, issue):
    if issue["disposition"] == "covered":
        return f"covered by node `{issue['covered_by']}`" + (f" -- {issue['reason']}" if issue["reason"] else "")
    if issue["disposition"] == "decline":
        return f"declined: {issue['reason']}"
    return "awaiting this node's decision"


def render_graph(data):
    """``final/<run>-graph.md``: the map every Last Order and review card reads."""
    lines = [f"# Research graph — run {data['run']}", "",
             "Written by misaka from the run's records after every change; edit nothing here, it is rewritten.",
             "", f"Question: {data['question']}", "",
             f"Every possibility raised so far, and where it went: `{runs.run_path({'id': data['run']}, PATHS_VIEW)}`.", "",
             (f"Status: {data['status']}/{data['phase']} · limits: depth {data['limits']['max_depth']}, "
              f"nodes {len(data['nodes'])}/{data['limits']['max_nodes']}, revisions {data['limits']['max_revisions']}, "
              f"follow-ups {data['limits']['max_followups']}"), "", "```mermaid", "flowchart TD"]
    by_id = {n["id"]: n for n in data["nodes"]}
    for node in data["nodes"]:
        lines.append(f'  {node["id"]}["{node["id"]} · {node["kind"]} · {node["status"]}<br/>{_mermaid_label(node["question"])}"]')
    for node in data["nodes"]:
        labels = {}
        for item in node["reached_by"]:
            labels.setdefault(item["decided_at"], []).append(item["label"])
        for parent in node["parents"]:
            label = "; ".join(labels.get(parent, []))
            arrow = "==>" if node["kind"] == "join" else "-->"
            lines.append(f"  {parent} {arrow}" + (f'|"{_mermaid_label(label)}"|' if label else "") + f" {node['id']}")
    for relation in data["relations"]:
        members = [m for m in relation["nodes"] if m in by_id]
        for a, b in pairwise(members):
            lines.append(f'  {a} -.-|"{relation["kind"]}"| {b}')
    lines += ["```", "", "## Nodes", "",
              "| Node | Kind | Depth | Status | Question | Parents | Reached by | Conclusion (latest) | Issues |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for node in data["nodes"]:
        reached = "; ".join(f"{r['decided_at']}: {r['label']}" for r in node["reached_by"]) or "-"
        conclusion = (node["files"].get("synthesis") or ["-"])[-1]
        if node["dissolves"]:
            conclusion += f" (dissolves {'the research question' if node['dissolves']['question'] == 'research_question' else 'its question'})"
        lines.append(f"| `{node['id']}` | {node['kind']} | {node['depth']} | {node['status']} | {_cell(node['question'])} "
                     f"| {', '.join(node['parents']) or '-'} | {_cell(reached)} | `{conclusion}` | {node['issues']} |")
    lines += ["", "## Decisions and options", ""]
    if not data["decisions"]:
        lines.append("No decision recorded yet.")
    for decision in data["decisions"]:
        lines += [f"### `{decision['node']}` · {_cell(decision['question'])} ({decision['origin']})", "",
                  f"Stakes: {_cell(decision['stakes'])}", "", "| Option | Premise | Outcome |", "| --- | --- | --- |"]
        lines += [f"| {_cell(o['label'])} | {_cell(o['premise'])} | {_cell(o['outcome'])} |" for o in decision["options"]]
        lines.append("")
    lines += ["## Relations", ""]
    lines += [f"- {r['kind']}: {', '.join(f'`{n}`' for n in r['nodes'])} — {_cell(r['note'])}" for r in data["relations"]] \
        or ["No relation recorded."]
    lines += ["", "## Reconciliations", ""]
    lines += [f"- round {r['round']}{'' if r['applied'] else ' (not applied yet)'}: {_cell(r['summary']) or '-'}"
              for r in data["reconciles"]] or ["None yet."]
    lines += ["", "## Corrections (errata)", ""]
    lines += [_erratum_line(e) for e in data["errata"]] or ["None recorded."]
    return "\n".join(lines) + "\n"


def _erratum_line(erratum):
    return (f"- `{erratum['node']}` says \"{_cell(erratum['claim'])}\" → {_cell(erratum['correction'])} "
            f"(recorded by `{erratum['raised_by']}`: {_cell(erratum['grounds'])})")


def render_paths(data):
    """``final/<run>-paths.md``: every possibility raised so far and where it went, under the node
    that raised it -- what a review reads before proposing, so nothing on it is raised again."""
    flat = lambda text: " ".join(str(text or "").split())
    lines = [f"# Paths — run {data['run']}", "",
             "Written by misaka from the run's records after every change; edit nothing here, it is rewritten.", "",
             ("Every possibility raised in this run -- each option of every decision, each divergence proposal -- "
              "under the node that raised it, with where it went. What is on this list is not raised again: a node "
              "named here pursues it, or it was declined or left unopened for the reason given."), ""]
    if not data["paths"]:
        lines.append("None raised yet.")
    for group in data["paths"]:
        lines += [f"## `{group['node']}` ({group['kind']}, depth {group['depth']}) — {_clip(group['question'], 120)}", ""]
        for entry in group["entries"]:
            if entry["what"] == "option":
                lines.append(f"- `{entry['id']}` option \"{flat(entry['text'])}\" (decision: {flat(entry['decision'])}) — "
                             f"premise: {flat(entry['premise'])} → {flat(entry['outcome'])}")
                lines += [f"  - takes up `{t['issue']}`: {flat(t['question'])}" for t in entry["takes_up"]]
            else:
                lines.append(f"- `{entry['id']}` proposal: {flat(entry['text'])} → {flat(entry['outcome'])}")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_node(data, node_id):
    """``nodes/<node>/NODE.md``: what this folder is, in the graph's words."""
    node = next(n for n in data["nodes"] if n["id"] == node_id)
    here = runs.node_dir(node_id)
    link = lambda path: f"`{posix_relpath(path, here)}`"
    lines = [f"# Node {node_id} — {_clip(node['question'], 120)}", "",
             "Written by misaka from the run's records; edit nothing here, it is rewritten.", "",
             f"- Run: `{data['run']}` · kind: {node['kind']} · depth {node['depth']} · status {node['status']}",
             f"- Question: {node['question']}",
             "- Parents: " + (", ".join(f"[`{p}`](../{p}/{NODE_VIEW})" for p in node["parents"]) or "none (the root)"),
             "- Children: " + (", ".join(f"[`{c}`](../{c}/{NODE_VIEW})" for c in node["children"]) or "none yet"),
             f"- Graph: `{posix_relpath(runs.run_path({'id': data['run']}, GRAPH_VIEW), here)}`"]
    if node["red_team"]:
        lines.append(f"- Red team and divergence review: Sister {node['red_team']['assignee']}"
                     + (f" — {node['red_team']['reason']}" if node["red_team"].get("reason") else ""))
    if node["dissolves"]:
        which = "the research question" if node["dissolves"]["question"] == "research_question" else "its own question"
        lines += ["", f"## Outcome: its conclusion dissolves {which}", "", node["dissolves"]["grounds"]]
    if node["reached_by"]:
        lines += ["", "## How this node was reached", ""]
        lines += [f"- at `{r['decided_at']}`, deciding \"{_clip(r['decision'], 100)}\": option \"{r['label']}\" — premise: {r['premise']}"
                  for r in node["reached_by"]]
    lines += ["", "## Files", ""]
    for name, paths in sorted(node["files"].items()):
        lines.append(f"- {name}: " + ", ".join(link(p) for p in paths))
    if not node["files"]:
        lines.append("- none yet")
    if node["cards"]:
        lines += ["", "## Cards", ""]
        lines += [f"- [{c['id']}] {_clip(c['title'], 80)} ({c['kind']}, round {c['round']}"
                  + (f", attempt {c['attempt']}" if c["attempt"] > 1 else "") + f", {c['status']})"
                  + (f" — {link(c['folder'])}" if c["folder"] else "") for c in node["cards"]]
    lines += ["", f"## Issues: {node['issues']}", "",
              f"The issues and their dispositions: misaka_research_view(view=\"issues\", node=\"{node_id}\").", ""]
    against = [e for e in data["errata"] if e["node"] == node_id]
    if against:
        lines += ["## Corrections recorded against this node", "", *map(_erratum_line, against), ""]
    made = [d for d in data["decisions"] if d["node"] == node_id]
    if made:
        lines += ["## Decisions made here", ""]
        for decision in made:
            lines.append(f"- {decision['question']} ({decision['origin']}) — stakes: {decision['stakes']}")
            lines += [f"  - {o['label']} — {o['premise']} → {o['outcome']}" for o in decision["options"]]
    return "\n".join(lines).rstrip() + "\n"


@contextmanager
def _writer(run_id):
    """One writer of a run's views at a time, across processes. The records are read inside the
    lock, so a later write never carries an older picture than an earlier one."""
    path = home.path("locks") / f"research-graph-{run_id}.lock"
    os.makedirs(path.parent, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        file_lock.lock_fd(fd)
        yield
    finally:
        os.close(fd)


def write_views(con, run, *, final=False):
    """Rewrite the graph view, the paths list and every node's NODE.md from the records; with ``final`` also the
    ``graph.json`` snapshot a finished (or halted) run leaves in the project. Derived output: a
    failure is logged and the workflow carries on. Never call this inside a database transaction."""
    try:
        with _writer(run["id"]):
            current = runs.get(con, run["id"])
            data = snapshot(con, current)
            workspace = current["workspace"]
            atomic.write_text(os.path.join(workspace, graph_path(current)), render_graph(data))
            atomic.write_text(os.path.join(workspace, paths_path(current)), render_paths(data))
            for node in data["nodes"]:
                folder = os.path.join(workspace, runs.node_dir(node["id"]))
                os.makedirs(folder, exist_ok=True)
                atomic.write_text(os.path.join(folder, NODE_VIEW), render_node(data, node["id"]))
            if final:
                atomic.write_text(os.path.join(workspace, runs.run_path(current, GRAPH_SNAPSHOT)),
                                  json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    except Exception as error:  # noqa: BLE001 - a view must never undo or stop the research it shows
        _LOG.warning("research %s: graph views could not be written: %s", run["id"], error)
