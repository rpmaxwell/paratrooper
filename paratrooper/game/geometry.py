"""Screen geometry shared by the whole package.

The game renders 320x200 and DOSBox-X doubles it into the 1024x768 Xvfb
screen. Starting the capture one row lower than the old scripts did
(y=201, not 200) makes every 2x2 screen block a single game pixel (100% of
blocks uniform over recorded frames), so we work on the native 320x200
grid: 4x fewer pixels, no information lost.

Coordinate systems:
  * native: (nx, ny) in 0..319 x 0..199 -- what perception works on.
  * game  : the old 640x400 "game area" coordinates every physics
    constant was measured and validated in (bomb_model, trooper_model,
    heli_model). game = (2*nx, 2*ny + 1). World/physics use game coords so
    the old, validated simulators remain ground truth for the new tables.
"""
SCREEN_X0, SCREEN_Y0 = 192, 201  # top-left of the game area in the Xvfb screen
SCREEN_W, SCREEN_H = 640, 400
NATIVE_W, NATIVE_H = 320, 200

BLACK, CYAN, MAGENTA, WHITE = 0, 1, 2, 3


def to_game(nx, ny):
    return 2 * nx, 2 * ny + 1


def to_native(gx, gy):
    return gx // 2, (gy - 1) // 2
