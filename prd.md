# PRD: Vision-Only RL Agent for DOS Paratrooper (1982)

## Background

This project is a proof-of-concept precursor to a larger effort: training an RL model to play F19 Strike Fighter (DOS) autonomously. That effort has stalled on reverse-engineering the game's memory layout via disassembly, DOSBox-X debugger traces, and Ghidra.

This PRD scopes down to a much simpler target — **Paratrooper (1982)** — to validate a different approach: an RL agent that observes **only the rendered screen** (pixels) and never reads emulator memory, for both observations and reward. If this works end-to-end on a simple game, the same architecture (screen capture → template-matched HUD reward → vision policy) can be ported to F19 without needing to finish the memory reverse-engineering work.

**Non-goal:** this PRD is not about F19. Do not add F19-specific code paths. Keep the environment wrapper generic enough that swapping games later is plausible, but don't over-engineer for that now.

## Goal

Produce a working Gym-style environment for Paratrooper, driven entirely by screen capture and synthetic key input, with a reward signal derived from on-screen score via template matching — then train a baseline RL agent (PPO or DQN) and confirm the reward curve trends upward over a short training run.

**Definition of done:** a random-action baseline and a trained baseline both run end-to-end without crashing, and the trained agent's average episode reward is measurably higher than the random agent's over the same number of episodes.

## Explicit constraints

- **No emulator memory reads of any kind** — not for observations, not for reward, not for the done signal. Everything must come from the rendered frame. This is the entire point of the POC; do not shortcut it by peeking at memory "just for now."
- Emulator: DOSBox-X, run headless under Xvfb.
- Language: Python.
- Keep each stage independently testable and validated before wiring it into the next. Do not build the full pipeline in one pass — see "Build sequence" below, which is a hard sequencing requirement, not a suggestion.

## Build sequence (build and validate in this order)

### Stage 0 — Emulator setup
- Get Paratrooper running manually in DOSBox-X first, confirmed working, before any automation.
- Disable sound. Set CPU cycles to a fixed value (not "auto") — auto-throttling will interfere with later fast-forwarding.
- Run DOSBox-X under Xvfb (virtual framebuffer), not a real display.
- **Validation gate:** confirm the game is visibly running correctly via a screenshot of the Xvfb display before proceeding.

### Stage 1 — Screen capture
- Use `mss` for frame capture (not shelling out to `import`/ImageMagick — too slow for RL throughput).
- Capture the Xvfb display region containing the game window.
- **Validation gate:** capture ~10 frames during manual play, save as PNGs, visually confirm frames are complete (not torn) and timing is consistent.

### Stage 2 — Action injection
- Use `xdotool` or `python-xlib` against the Xvfb `DISPLAY` to send key events.
- Confirm actual Paratrooper key bindings for the specific version/build in use (don't assume).
- Discrete action space, minimum viable set:
  - `0` = noop
  - `1` = rotate turret left
  - `2` = rotate turret right
  - `3` = fire
  - (Add left+fire / right+fire combos later only if needed — confirm whether the turret can fire while rotating first.)
- **Validation gate:** script a fixed, non-RL action sequence, screenshot before/after each action, manually confirm the turret responded correctly. This isolates key-timing bugs before they contaminate training data.

### Stage 3 — Reward extraction (score reading)
- Paratrooper's score is a fixed-position, monospaced bitmap font on a small fixed palette — use template matching, not general OCR.
- Steps:
  1. Determine the fixed pixel rectangle for the score region by inspection.
  2. Capture one clean reference template per digit (0–9) as rendered in-game.
  3. Split the cropped score region into fixed-width character cells; match each cell to the closest digit template (pixel-diff or normalized cross-correlation).
  4. Concatenate matched digits into an integer score.
- Also template-match:
  - Remaining-lives/cannons indicator (if present on screen).
  - A distinct "game over" screen state — this is the `done` signal.
- Handle score digit-count changes (e.g., single digit → double digit) via fixed-width, right-aligned (or left-padded) digit fields so cell alignment doesn't break when the score grows.
- **Validation gate:** play manually for several minutes while logging `read_score()` every frame; confirm logged values match observed on-screen score exactly, including transitions across digit-count boundaries and correct triggering of the `done` signal on game over.

### Stage 4 — Gym-style environment wrapper
- Only begin once Stages 1–3 are each independently validated.
- Implement a `gymnasium.Env` subclass:
  - `action_space`: `Discrete(4)` (per Stage 2).
  - `observation_space`: grayscale, downsampled frame (e.g. 84×84), optionally frame-stacked (e.g. 4 frames) to convey motion/fall direction, following Atari-RL convention.
  - `reset()`: restarts the game (keypress-triggered restart or DOSBox save-state reload), resets internal score tracking, returns initial processed observation.
  - `step(action)`: sends action, captures frame, computes reward as score delta since last step, checks `done` via game-over template match, returns `(obs, reward, done, truncated, info)`.
- **Validation gate:** run the environment with a **random-action agent** for several hundred steps before introducing any RL algorithm. Confirm:
  - Reward is nonzero at expected moments (e.g., a kill).
  - `done` fires correctly on death.
  - `reset()` produces a clean state with no leakage from the previous episode.
  - This step is required — most "RL isn't learning" failures trace back to a broken environment, not bad hyperparameters, and are far cheaper to catch here than after training.

### Stage 5 — Throughput / speed
- Real-time DOSBox-X speed is too slow for RL sample requirements. In order of effort:
  1. Raise DOSBox-X `cycles` to a fixed, higher value. First confirm Paratrooper's game logic isn't wall-clock-locked independent of CPU cycles (some early-80s titles are — if so, this approach has a ceiling).
  2. Run multiple parallel DOSBox-X + Xvfb instances (vectorized environment) rather than trying to push a single instance faster — better returns than over-optimizing one instance.

### Stage 6 — Baseline training run
- Only after Stage 4's random-agent validation gate passes.
- Use `stable-baselines3` (PPO or DQN) with a small CNN policy on the wrapped environment.
- No hyperparameter tuning in this pass — the only goal is confirming the reward curve trends upward over a few thousand steps relative to the random baseline.
- Once this is confirmed, this PRD's scope is complete. Reward shaping, frame-stack tuning, and speed optimization are follow-on work, not part of this POC.

## Out of scope for this POC
- Any F19-specific code or assumptions.
- Reward shaping beyond raw score delta.
- Hyperparameter tuning.
- Multi-instance/vectorized training infrastructure beyond a basic proof it's possible (Stage 5.2 is "confirm feasible," not "build a production training cluster").

## Open questions to confirm during implementation
- Exact key bindings for the specific Paratrooper build being used.
- Whether the turret can rotate and fire simultaneously (affects action space design).
- Whether Paratrooper's game loop is wall-clock-timed or cycle-timed (affects how much Stage 5.1 speedup is achievable).
- Exact screen coordinates for the score region and game-over screen (must be determined by inspection of the actual running game, not assumed).
