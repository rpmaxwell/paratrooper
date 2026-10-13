# paratrooper

A bot that plays Paratrooper (1982) in DOSBox-X: it reads the game from the
screen or its memory, plans shots with the game's own physics, and records
every game for later analysis.

- Layout, commands and working rules: [CLAUDE.md](CLAUDE.md)
- How the bot works: [docs/process.md](docs/process.md)
- How it got here: [docs/dev_history.md](docs/dev_history.md)

Quick start (needs Docker and the game file at
`~/Documents/paratrooper/ParaTrooper.1982.com`, or `PARATROOPER_COM`):

```
uv venv && uv pip install -e '.[analysis,dev]'
echo 'VNC_PASSWORD=...' > .env
python -m paratrooper up
python -m paratrooper play configs/base.toml --games 3
```
