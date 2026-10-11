# durable

A small local coding agent on `misaka.durable`. One process owns the model runtime, the durable Harness, its SQLite storage, and the TUI. It reuses MISAKA's model runtime, auth, settings, system prompt, keybindings, theme, and interactive components; the agent itself is the durable Harness with its built-in `CodingTools`.

```bash
python -m misaka.experimental.durable.main
python -m misaka.experimental.durable.main --continue
```

A new session starts with MISAKA's default model and thinking level from `settings.json`. `--continue` opens the newest session for the current directory. Sessions live under `state/experimental/durable-sessions/<cwd-hash>/<session>/session.sqlite` in the MISAKA home; a lock keeps a second process out (a lock left by a crash goes stale after 10 seconds, and the next start waits for that). Log in with MISAKA itself; credentials are shared.

## What it shows

- **Durability:** every streamed partial, tool output, queue, and turn is committed. Kill the process in the middle of a tool call and start it again with `--continue`: the interrupted call gets an interrupted result and the turn finishes. Nothing in the TUI handles recovery; it only renders the conversation view.
- **One view:** the TUI renders `Conversation.viewState()`, the structural mount of the transcript and the built-in documents (`pi.live`, `pi.inbox`, `pi.agent`, `pi.usage`). Streaming, tool progress, the queue, retries, compaction status, the model, and usage all come from it.
- **Subagents:** the `subagent` tool runs a task in a child conversation owned by the call. `/agents` switches the view to any conversation, also while the subagent works, and the editor then talks to it: steer it while busy, or keep chatting after the call returned. Esc aborts the shown conversation; aborting the main turn aborts its subagents.
- **Task graph:** a live panel of `Harness.taskGraph()`, shown by default and toggled with `/tasks`: every live task, what it waits on, and the conversations it owns.

## Commands

- submit: prompt while idle, steer while busy
- follow-up key (`app.message.followUp`): queue a follow-up
- Esc: abort the shown conversation's work, including a manual compaction
- `/model` or the model key: select a model for the shown conversation
- thinking key (Shift+Tab): cycle thinking levels
- `/compact [instructions]`: summarize older context; reports "Nothing to compact" when the context fits in `compaction.keepRecentTokens`
- `/agents`: switch conversations
- `/tasks`: hide or show the task panel
- tools expand key (Ctrl+O): expand tool output and compaction summaries
- clear key (Ctrl+C) or Ctrl+D: exit at once, unlike MISAKA's clear-first Ctrl+C; work in flight resumes with `--continue`

## Layout

| file | role |
| --- | --- |
| `main.py` | arguments, open, run, close |
| `sessions.py` | session directories and the lock |
| `runtime.py` | Harness, registry, settings, environments; the plain `DurableView` and `DurableController` |
| `prompt.py` | MISAKA's system prompt sections (tools, rules, project context, cwd) as one extension |
| `harness_setup.py` | the Harness's settings, tool registry and execution environments, shared with the experimental Session worker |
| `subagent.py` | the foreground subagent tool |
| `tui.py` | rendering with MISAKA's interactive components |

Compaction thresholds, retry policy, queue modes, and request timeouts come from MISAKA's `settings.json` as loaded at startup, read through Harness settings getters at each use.

A turn that ends without an answer shows a notice; one recovered after a restart does not, since only submissions made by this process are watched.

The demo runs in MISAKA's regular terminal renderer: the transcript, the task panel, the queue, the notices, the working status, the editor, and the footer are stacked in that order and the terminal scrolls. It has no fullscreen mode.

Not here: sessions list and resume picker, forks and tree navigation, extensions, prompt templates, images, `/login`.
