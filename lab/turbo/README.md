# Turbo: can the game run faster than real time?

**Status: settled (2026-10-13).** Yes. DOSBox-X turbo lets emulated time run
as fast as the host allows. The game paces itself by busy-waiting on the
BIOS tick (INT 1Ah), which follows emulated time, so `cycles` becomes the
throttle. Raising cycles without turbo never sped the game up.

Measured with `measure_turbo.py` (one container, arm64 Mac, no player input):

| cycles | speed vs real time | game tick vs BIOS tick |
|---|---|---|
| 3000 | ~20x | exact |
| 1000 | ~46x | exact |
| 300 | ~85-90x | exact (safe floor; ~ a 4.77 MHz 8088) |
| 100 | ~95x | lags and catches up (jitter) |
| 50 | ~115x | drops ticks |

`stop turbo on key = false` is required (the default ends turbo at the first
keypress).

Run it: `python -m paratrooper up --speed turbo --cycles 300 --instance 1`,
then `docker exec paratrooper-1 python3 /repo/lab/turbo/measure_turbo.py 20`.

**What it means for the bot:** the emulator side is in
`paratrooper/emulator/docker.py` (`speed = "turbo"`). The bot can't play under
turbo yet: its control loop, tracking timeouts and the vision tick clock
count wall-clock seconds, and under turbo a second is 20-90 game ticks.
`config.validate` refuses `play` with turbo until they count game ticks.
At speed S the bot has 55/S ms per tick (2.75 ms at 20x), so beyond ~20x
the emulator has to wait for the bot each tick (lockstep).
