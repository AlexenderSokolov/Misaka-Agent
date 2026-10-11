# Running MISAKA safely

Treat what the models do as untrusted. Last Order, the Sisters and their sub-agents read, change
and run files with the permissions of your account, and they do not ask before every tool call.
Extensions, MCP servers and the programs a command starts run with the same permissions. This is
pi's design, which MISAKA keeps; its wording is pi's (`docs/security.md`).

Pages, documents, card results and command output can steer a model through prompt injection, and
research reads a great deal of all of them. Watching the panel, reviewing cards and plan approval
help you notice; they are not a security boundary.

## What the project folder does and does not do

The project folder decides where tools work by default and what a run's report counts as its
sources. It does not stop a tool or a command from reaching other paths your account can read or
write.

What MISAKA does keep out of the models' way, as defence in depth (a shell command can still
reach all of it):

- the live skill folders, which change only through `skill_manage`;
- MISAKA's own credentials, `~/.misaka/credentials/` and the `.env` of the home and of its roles,
  which the file tools (`read`, `grep`, `find`) do not open, in any spelling of the path;
- the files every later session loads as its instructions -- `PROJECT.md`, `AGENTS.md`,
  `CLAUDE.md` in the project or a folder above it -- which a Sister or a sub-agent does not
  change (Last Order and you can);
- `~/.misaka` itself, which an unattended session (a card, a sub-agent) does not write into;
- API keys, which an MCP server does not inherit: it gets a small part of the environment (paths,
  locale, proxy and certificate settings) and the variables it is configured with.

What MISAKA itself passes to a model from outside -- fetched pages, document pages, findings, a
Sister's output, the materials listed in a card -- is fenced as data, not instructions. What a
model reads on its own with `read` or a command is not fenced.

## Instruction files above the project

As in pi, a session loads `PROJECT.md`, `AGENTS.md` or `CLAUDE.md` from its folder and from every
folder above it. Keep such files out of folders you share or download into, such as `/tmp`, your
home folder or a shared drive, or start MISAKA from a folder whose parents you control.

## Running it more safely

Only give MISAKA the files and services a task needs. Run it as a dedicated user, or inside a
container or virtual machine, when the material you research is not yours. Keep credentials you do
not want a model to reach outside that environment, or use narrowly scoped ones.
