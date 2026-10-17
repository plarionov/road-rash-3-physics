#!/usr/bin/env python3
"""Assemble the fixed-timestep hooks and write a ROM plus an IPS patch.

Time base is 1/300 second, so 50 Hz and 60 Hz share one step size:
  NTSC vblank = 5 units (1/60 s)
  PAL  vblank = 6 units (1/50 s)
  one physics step = 15 units (1/20 s)
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORG = 0x1FA0C0

# Filled by configure(). Defaults: ROM and Ti folder next to this script.
SRC = None
OUT_DIR = None
OUT_ROM = None
OUT_IPS = None
TI_DIR = None
VARIANTS = None


def configure(rom=None, out=None, ti_dir=None):
    global SRC, OUT_DIR, OUT_ROM, OUT_IPS, TI_DIR, VARIANTS
    OUT_DIR = Path(out).resolve() if out else HERE
    SRC = Path(rom).resolve() if rom else OUT_DIR / "Road Rash 3 (USA, Europe).gen"
    TI_DIR = Path(ti_dir).resolve() if ti_dir else OUT_DIR / "ROAD_RASH_3_Improvement_v1.04"
    OUT_ROM = OUT_DIR / "Road Rash 3 (physics).gen"
    OUT_IPS = OUT_DIR / "road-rash-3-physics.ips"
    # After the standard IPS the first long $FF run starts at $1FCB44.
    # Special fills that run; its own free tail starts at $1FDC6A.
    VARIANTS = {
        "ti": (0x1FCB44, TI_DIR / "ROAD_RASH_3_Improvement_v1.04.ips",
               OUT_DIR / "Road Rash 3 (physics+ti).gen",
               OUT_DIR / "road-rash-3-physics-ti.ips"),
        "special": (0x1FDC6A, TI_DIR / "ROAD_RASH_3_Improvement_v1.04_Special.ips",
                    OUT_DIR / "Road Rash 3 (physics+ti-special).gen",
                    OUT_DIR / "road-rash-3-physics-ti-special.ips"),
    }
STEP = 15
CLAMP = 30
DELTA_CAP = 8

# The sound driver owns FF0000-FF01F1, including the byte queue that a
# skid command is written into. Parking the step clock there stuck that
# sound until a pause rebuilt the queue, and it also dropped clock ticks.
# FF01FF is a game flag, so the gap between them is FF01F2-FF01FE.
ACCUM = 0xFF01F2
VBL = 0xFF01F4
LAST = 0xFF01F6
INIT = 0xFF01F8
STEPF = 0xFF01F9
INDONE = 0xFF01FA
NEED = 0xFF01FB

# Base-patch clock. FF01FF stays a game flag, so this block is 13 bytes.
# A 4.4 fixed ratio of 0x10 is stock speed. NTSC scanlines are 262.
RATIO = 0xFF01F2          # byte, 0 until calibrated, then >= $10
ENTRY = 0xFF01F3          # raw V counter, written after registers are saved
HAND = 0xFF01F6           # word, scanlines spent inside the vblank handler
OWED = 0xFF01F8           # word, stock main-thread lines the step must cover
SNAP_VBL = 0xFF01FA
SNAP_HAND = 0xFF01FC
SNAP_V = 0xFF01FE         # raw vertical counter byte
# FF02C2-FF036D stay zero through vanilla races, solo and split.
# EST16 averages the 100% frame length in 1/16 vblank; FRAME16 is what
# the step uses, 0 when the original step runs.
LAT_REM = 0xFF0300        # 16 rider words, then one spare
EST16 = 0xFF02D0
FRAME16 = 0xFF02D4
HILL_REM = 0xFF02D6       # two words, one per mountain site
CAL_RAW = 0xFF02DC        # long, last calibration count
# Solo 100% frame for the Ti builds, in 1/16 vblank. Stock is about 7.
TI_SOLO_FRAME16 = 128
# FF02F8-FF02FF have no absolute references in the game. The rider counter
# at FF0266 indexes at most eight slots and stops near FF02DE. The old
# cookie lived at FF03F8, the first pointer of the far-behind list, so a
# rider dropping back restarted the measure in the middle of a race.
RECHECK = 0xFF02FC
IN_VBL = 0xFF02FD           # set for the whole vblank callback
CALIB_HOLD = 0xFF02F8       # bit 0: Up+C is held; bit 1: backdrop is ours
CALIB_MARK = 0xFF02F9       # announcement frame, 0 when idle
CALIB_COLOR = 0xFF02FA
# Remembered upward speed of the player while a climb is held on the road.
# 50(a1) is copied into the shadow, so the speed cannot live there.
KICK = 0xFF02FA      # word, backdrop saved for the announcement
# Road scratch, rewritten every build. STEP is the slope the next row
# still has to climb. FIX is added once to the foreground raw height so
# the rider stays on the original curve; the polygon itself is not moved.
ROAD_FIX = 0xFF02F2
ROAD_STEP = 0xFF02FE
# Second rider's own camera offset. The stock word is shared, so player 2's
# view was shifted by whatever player 1 had just added.
P2_CAM = 0xFF02F4
# Low 8 bits of speed*frame that the heading shift would throw away.
# One byte per rider slot, indexes 0..11 from FF09B2. The results table
# ends at FF02E2, and FF02F0 is the camera hold, so this sits between them.
# One finished 2P picture is waiting for the upload. The buffer flag looks
# free as soon as the two buffers swap, so a fast CPU starts the next camera
# step while the interrupt is still reading the previous one. FF02F0 has no
# absolute reference in the game. Stock (ratio $90) never sets it.
CAM_HOLD = 0xFF02F0
# Set while the two-frame sample is spinning. The vblank then only counts
# itself: the split-screen upload is a long CPU copy, and counting through
# it records free time instead of the clock. 300% was coming out one flash
# short. FF02F1 has no absolute reference in the game.
CALIB_SPIN = 0xFF02F1
# Iterations of the sample loop in two vblanks at 7.67 MHz while the game
# callback is not running. The old 2491 included that callback, so a
# two-player frame, which spends the vblank copying both pictures, read
# about one step low. Measured 3490-3580 per 100% at 100-400%, in the
# race and from the pause. 14..17 in the 4.4 ratio collapses back to stock.
STOCK_LINES = 3530
PHYS_DONE_AT = 0x1FA226


class Asm:
    def __init__(self, org):
        self.org = org
        self.here = org
        self.buf = bytearray()
        self.labels = {}
        self.fixups = []

    def label(self, name):
        self.labels[name] = self.here

    def emit(self, data):
        self.buf += data
        self.here += len(data)

    def w(self, value):
        self.emit((value & 0xFFFF).to_bytes(2, "big"))

    def l(self, value):
        self.emit((value & 0xFFFFFFFF).to_bytes(4, "big"))

    def abs_w(self, opcode, addr):
        self.w(opcode)
        self.l(addr)

    def bcc(self, cond, name):
        # Bcc.w. Displacement is relative to the extension word.
        self.w(cond << 8)
        at = len(self.buf)
        self.w(0)
        self.fixups.append((at, name, self.org + at))
        self.here += 0  # w() already advanced here

    def dbf(self, reg, name):
        # DBcc always carries a 16-bit displacement from the extension word.
        self.w(0x51C8 + reg)
        at = len(self.buf)
        self.w(0)
        self.fixups.append((at, name, self.org + at))

    def finish(self):
        for at, name, pc_of_disp in self.fixups:
            target = self.labels[name]
            disp = (target - pc_of_disp) & 0xFFFF
            self.buf[at:at + 2] = disp.to_bytes(2, "big")
        return bytes(self.buf)


def build_code(org, ti=False):
    a = Asm(org)

    # ---- vblank entry, replaces 10 bytes at 0x21B6 ----
    # Count the interrupt the CPU actually took, then run the original handler.
    # A short-circuit for "already inside vblank" freezes the EA logo: the
    # handler later drops IPL to 3, the pending interrupt re-enters, and that
    # re-entry never returns to the game.
    a.label("vbl_entry")
    a.abs_w(0x5279, VBL)             # addq.w #1, vbl_counter
    a.w(0x48E7)
    a.w(0xFFFE)                      # movem.l d0-d7/a0-a6, -(sp)
    if ti:
        # Ti inserts the Sega CD bus request immediately after the movem.
        a.w(0x08F9)
        a.w(0x0008)
        a.l(0xA12000)                # bset #8, $A12000
        a.w(0x4EF9)
        a.l(0x21C2)
    else:
        a.abs_w(0x2079, 0xFF0226)    # movea.l callback, a0
        a.w(0x4EF9)
        a.l(0x21C0)                  # jmp original handler

    # ---- start of the race frame, jsr from 0x12D7C ----
    a.label("frame_start")
    if not ti:
        a.w(0x33FC)
        a.w(0x8134)
        a.l(0xC00004)                # display off, original first instruction
    a.abs_w(0x3239, 0xC00004)        # d1 = VDP status
    a.w(0x0801)
    a.w(0x0000)                      # btst #0, d1
    a.w(0x56C2)                      # sne d2  (PAL -> $FF)
    # A loaded savestate can hold $FF here. Only a marker we wrote counts.
    a.w(0x0C39)
    a.w(0x00A5)
    a.l(INIT)                        # cmpi.b #$A5, init
    a.bcc(0x67, "inited")
    a.w(0x13FC)
    a.w(0x00A5)
    a.l(INIT)                        # move.b #$A5, init
    a.abs_w(0x4279, ACCUM)           # clr.w accum
    a.abs_w(0x4239, STEPF)
    a.abs_w(0x4239, NEED)
    a.w(0x33F9)
    a.l(VBL)
    a.l(LAST)                        # last = counter, drop the pre-race gap
    a.w(0x7001)                      # moveq #1, d0
    a.bcc(0x60, "got_delta")
    a.label("inited")
    a.abs_w(0x3239, LAST)            # move.w last, d1
    a.abs_w(0x3039, VBL)             # move.w counter, d0
    a.w(0x33C0)
    a.l(LAST)                        # last = snapshot, a nested tick stays for next time
    a.w(0x9041)                      # sub.w d1, d0
    a.w(0x0C40)
    a.w(DELTA_CAP)                   # cmpi.w #8, d0
    a.bcc(0x63, "delta_ok")          # bls
    a.w(0x7001)                      # a pause between races is one frame
    a.label("delta_ok")
    a.label("got_delta")
    a.w(0x4A40)                      # tst.w d0
    a.bcc(0x67, "no_time")           # beq
    a.w(0x7200)                      # moveq #0, d1
    a.w(0x3200)                      # move.w d0, d1
    a.w(0x4A02)                      # tst.b d2
    a.bcc(0x66, "pal_units")
    a.w(0xC2FC)
    a.w(5)                           # mulu.w #5, d1
    a.bcc(0x60, "add_units")
    a.label("pal_units")
    a.w(0xC2FC)
    a.w(6)                           # mulu.w #6, d1
    a.label("add_units")
    a.abs_w(0xD379, ACCUM)           # add.w d1, accum
    a.label("no_time")
    a.abs_w(0x3039, ACCUM)
    a.w(0x0C40)
    a.w(CLAMP)                       # cmpi.w #30, d0
    a.bcc(0x63, "clamped")           # bls
    a.w(0x33FC)
    a.w(CLAMP)
    a.l(ACCUM)
    a.label("clamped")
    a.abs_w(0x4A39, STEPF)           # tst.b step_pending
    a.w(0x4E71)
    a.w(0x4E71)                      # arm a step on every finished frame
    a.abs_w(0x3039, ACCUM)
    a.w(0x0C40)
    a.w(STEP)
    a.w(0x4E71)
    a.w(0x4E71)
    a.w(0x0479)
    a.w(STEP)
    a.l(ACCUM)                       # subi.w #15, accum
    a.abs_w(0x50F9, STEPF)           # st step_pending
    a.abs_w(0x4239, INDONE)          # clr.b input_done
    a.abs_w(0x50F9, NEED)            # one pad sample is due for this step
    a.label("frame_done")
    if ti:
        a.w(0x4DF9)
        a.l(0xFF0000)                # lea $FF0000, a6  (the instruction we replaced)
    a.w(0x4E75)                      # rts

    if ti:
        # Ti inlined the HUD clock. Tick it once per physics step, the same
        # rate a finished frame has without overclock, then leave a0 on the
        # clock word for the rest of his routine.
        a.label("clock_tick")
        a.w(0x41EE)
        a.w(0x09A0)                  # lea $9A0(a6), a0
        a.abs_w(0x4A39, STEPF)       # kept so this site stays put
        a.w(0x4E71)
        a.w(0x4E71)                  # the HUD second follows the frame, as in the unpatched game
        a.abs_w(0x4A39, INDONE)
        a.w(0x4E71)
        a.w(0x4E71)
        a.abs_w(0x50F9, INDONE)
        a.w(0x5250)                  # addq.w #1, (a0)
        a.label("clock_done")
        a.w(0x4E75)
    # The stick and the clock run on every frame callback. Gating them to the
    # physics step left the lean at about a third of the stock value: after
    # 2.2 s of left the unpatched bike was on the stop (-2048) and these
    # builds were near -650. The movement step below stays on the 50 ms clock.
    a.label("input_gate")
    a.abs_w(0x4A39, NEED)
    a.w(0x4E71)
    a.w(0x4E71)                      # was beq input_skip
    a.abs_w(0x4239, NEED)
    if not ti:
        a.w(0x4EB9)
        a.l(0x12CA8)                 # HUD clock, once per step
    a.abs_w(0x4A79, 0xFF2534)
    a.w(0x4E75)
    a.label("input_skip")
    a.w(0x588F)                      # addq.l #4, sp
    a.w(0x4EF9)
    a.l(0x13038)

    # ---- physics gate, jsr from 0x13D76 ----
    # Run the body whenever the render buffer is free, once per finished
    # frame, which is what the stock game does. The 50 ms gate left the
    # skid decay (a quarter off, once per body call) running about ten
    # times a second while this course already finishes a frame every vblank.
    a.label("phys_gate")
    a.w(0x0CAF)
    a.l(0x185F86)
    a.w(4)                           # cmpi.l #$185F86, 4(sp)
    a.bcc(0x66, "phys_setup")
    a.abs_w(0x4A39, STEPF)
    a.bcc(0x67, "phys_skip")         # at most one body call per frame
    a.abs_w(0x2079, 0xFF278A)        # a0 = buffer flag
    a.w(0x4A50)                      # tst.w (a0)
    a.bcc(0x66, "phys_skip")         # buffer not consumed yet
    a.w(0x2EBC)                      # move.l #phys_done, (sp)
    a.l(0)
    done_at = len(a.buf) - 4
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4EF9)
    a.l(0x13D7C)
    a.label("phys_setup")
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4A50)
    a.bcc(0x66, "phys_skip")
    a.w(0x2EBC)                      # move.l #setup_done, (sp)
    a.l(0)
    setup_at = len(a.buf) - 4
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4EF9)
    a.l(0x13D7C)
    a.label("phys_skip")
    a.w(0x588F)
    a.w(0x4E75)
    a.label("phys_done")
    a.w(0x4EB9)
    count_at = len(a.buf)
    a.l(0)                           # jsr count_step, same size as clr.b
    a.w(0x4E75)
    a.label("setup_done")
    a.w(0x4E75)
    a.label("count_step")
    a.abs_w(0x4239, STEPF)
    a.w(0x5239)
    a.l(0xFF01FE)                    # steps completed, one byte
    a.w(0x4E75)
    a.buf[done_at:done_at + 4] = a.labels["phys_done"].to_bytes(4, "big")
    a.buf[setup_at:setup_at + 4] = a.labels["setup_done"].to_bytes(4, "big")
    a.buf[count_at:count_at + 4] = a.labels["count_step"].to_bytes(4, "big")

    code = a.finish()
    if ORG + len(code) > 0x200000:
        raise SystemExit("patch does not fit")
    return code, a.labels


def header_checksum(rom):
    """Sega header word at 0x18E: sum of big-endian words from 0x200 to EOF."""
    total = 0
    for i in range(0x200, len(rom), 2):
        total = (total + int.from_bytes(rom[i:i + 2], "big")) & 0xFFFF
    return total


def sega_checksum(rom):
    """Same sum the game runs at 0x1FA064."""
    total = 0
    addr = 0
    for _ in range(0x7E819):
        if addr == 0x18C:
            addr += 4
        else:
            total = (total + int.from_bytes(rom[addr:addr + 4], "big")) & 0xFFFFFFFF
            addr += 4
    return total


def apply_ips(rom, path):
    data = Path(path).read_bytes()
    if data[:5] != b"PATCH" or data[-3:] != b"EOF":
        raise SystemExit(f"not an IPS: {path}")
    i = 5
    while i < len(data) - 3:
        off = int.from_bytes(data[i:i + 3], "big")
        i += 3
        ln = int.from_bytes(data[i:i + 2], "big")
        i += 2
        if ln == 0:
            rle = int.from_bytes(data[i:i + 2], "big")
            i += 2
            rom[off:off + rle] = bytes([data[i]]) * rle
            i += 1
        else:
            rom[off:off + ln] = data[i:i + ln]
            i += ln


def ips_record(offset, data):
    if offset > 0xFFFFFF:
        raise SystemExit(f"offset too big for IPS: {offset:#x}")
    out = offset.to_bytes(3, "big")
    out += len(data).to_bytes(2, "big")
    out += data
    return out


def write_ips(original, rom, path):
    ranges = []
    start = None
    for i, (a_b, b_b) in enumerate(zip(original, rom)):
        if a_b != b_b:
            if start is None:
                start = i
            end = i + 1
        elif start is not None:
            ranges.append((start, end))
            start = None
    if start is not None:
        ranges.append((start, len(rom)))
    ips = bytearray(b"PATCH")
    for start, end in ranges:
        ips += ips_record(start, bytes(rom[start:end]))
    ips += b"EOF"
    path.write_bytes(ips)
    return len(ranges), len(ips)


def build_base(org, cd=False, fps_tick=False):
    """Keep the original once-per-frame step, and stretch it when the CPU
    finishes the main-thread work sooner than a 7.67 MHz machine would.

    The render slices already take the same real time at any clock. Only the
    main-thread part of $13D76 scales with overclock, so the next step waits
    until the time since this step started covers that work at stock speed.
    """
    a = Asm(org)

    a.label("vbl_entry")
    # The interrupted instruction's registers have to be on the stack before
    # anything here uses d0. Saving them afterwards put the scanline into the
    # slot the game expected to get back, and the race loop never returned.
    a.w(0x48E7)
    a.w(0xFFFE)                      # movem.l d0-d7/a0-a6, -(sp)
    if fps_tick:
        # Ti's frame counts vblanks here and the FPS drawer samples that
        # flag. This build puts the stock frame back, so the count lives
        # on the vblank entry instead.
        a.abs_w(0x5339, 0xFF1BE7)    # subq.b #1, period
        a.bcc(0x66, "fps_period")
        a.w(0x33FC)
        a.w(0x013C)
        a.l(0xFF1BE6)                # move.w #$13C, period
        a.label("fps_period")
    a.abs_w(0x5279, VBL)             # addq.w #1, vbl
    a.w(0x4A39)
    a.l(CALIB_SPIN)                  # tst.b spin
    a.bcc(0x66, "vbl_fast")          # the sample is counting this vblank
    a.w(0x1039)
    a.l(0xC00008)                    # move.b HV, d0  (V is the high byte)
    a.w(0x13C0)
    a.l(ENTRY)                       # move.b d0, entry
    a.abs_w(0x2079, 0xFF0226)        # movea.l callback, a0
    a.w(0x08F9)
    a.w(0x0000)
    a.l(IN_VBL)                      # bset #0, in_vbl
    if cd:
        # Ti requests the Sega CD bus here and then calls the frame
        # pointer himself. $21C0 sits inside that request. $21CE is the
        # first instruction after his call.
        a.w(0x08F9)
        a.w(0x0008)
        a.l(0xA12000)                # bset #8, $A12000
        a.w(0xB1FC)
        a.l(0)                       # cmpa.l #0, a0
        a.bcc(0x67, "vbl_nocall")
        a.w(0x4E90)                  # jsr (a0)
        a.label("vbl_nocall")
        a.w(0x4EF9)
        a.l(0x21CE)
    else:
        a.w(0x4EF9)
        a.l(0x21C0)

    # Replaces the movem/rte at $221A. IPL was lowered to 3 by the handler,
    # so a second vblank can land while this exit is still reading the counter.
    # Sample vblank. The registers were saved on entry. Skipping the game
    # callback keeps the two-player copy out of the clock measurement.
    a.label("vbl_fast")
    if cd:
        a.w(0x08F9)
        a.w(0x0008)
        a.l(0xA12000)                # bset #8, $A12000
    a.w(0x4CDF)
    a.w(0x7FFF)                      # movem.l (sp)+, d0-d7/a0-a6
    a.w(0x4E73)                      # rte

    a.label("vbl_leave")
    a.w(0x08B9)
    a.w(0x0000)
    a.l(IN_VBL)                      # bclr #0, in_vbl
    a.w(0x007C)
    a.w(0x0700)                      # ori.w #$0700, sr
    a.w(0x7000)                      # moveq #0, d0
    a.w(0x1039)
    a.l(ENTRY)                       # move.b entry, d0
    a.bcc(0x61, "vline")
    a.w(0x3400)                      # move.w d0, d2
    a.abs_w(0x3039, 0xC00008)
    a.w(0xE048)                      # lsr.w #8, d0
    a.bcc(0x61, "vline")
    a.w(0x9042)                      # sub.w d2, d0
    a.bcc(0x64, "hand_pos")          # bcc: delta did not wrap
    a.w(0x0640)
    a.w(262)                         # addi.w #262, d0
    a.label("hand_pos")
    a.abs_w(0xD179, HAND)            # add.w d0, hand
    a.bcc(0x61, "show_cal")          # bsr: backdrop, after the game's own vblank
    a.w(0x4CDF)
    a.w(0x7FFF)                      # movem.l (sp)+, d0-d7/a0-a6
    a.w(0x4E73)                      # rte

    a.label("phys_gate")
    a.w(0x0CAF)
    a.l(0x185F86)
    a.w(4)                           # cmpi.l #$185F86, 4(sp)
    a.bcc(0x66, "phys_setup")
    a.w(0x4EB9)
    calib_at = len(a.buf)
    a.l(0)                           # jsr calib
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4A50)                      # tst.w (a0)
    a.bcc(0x66, "phys_skip")
    a.w(0x4A39)
    a.l(CAM_HOLD)                    # tst.b hold
    a.bcc(0x66, "phys_skip")         # picture not taken yet
    # Snap holds the calibration timestamp until bit 7 says the timing is
    # committed. Absorbing before that would spin on the timestamp itself.
    a.abs_w(0x1039, RATIO)
    a.w(0x0800)
    a.w(0x0007)                      # btst #7, d0
    a.bcc(0x67, "phys_body")         # beq
    if STOCK_LINES:
        # How much of a stock step this frame represents. The body always
        # runs, so the picture is not held up; the skid delta uses the fraction.
        a.w(0x4EB9)
        scale_at = len(a.buf)
        a.l(0)
    a.w(0x4EB9)
    snap_at = len(a.buf)
    a.l(0)
    a.label("phys_body")
    a.w(0x4EB9)
    est_at = len(a.buf)
    a.l(0)                           # jsr est_update
    a.w(0x2EBC)
    done_at = org + (PHYS_DONE_AT - ORG)
    a.l(done_at)                     # move.l #phys_done, (sp)
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4EF9)
    a.l(0x13D7C)
    a.label("phys_setup")
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4A50)
    a.bcc(0x66, "phys_skip")
    a.w(0x2EBC)
    a.l(0)
    setup_at = len(a.buf) - 4
    a.abs_w(0x2079, 0xFF278A)
    a.w(0x4EF9)
    a.l(0x13D7C)
    a.label("phys_skip")
    a.w(0x588F)                      # addq.l #4, sp
    a.w(0x4E75)

    if a.here > done_at:
        raise SystemExit(f"phys_done moved past the save slot: {a.here:#x}")
    while a.here < done_at:
        a.w(0x4E71)
    a.label("phys_done")
    if STOCK_LINES:
        a.w(0x4EB9)
        charge_at = len(a.buf)
        a.l(0)
    a.w(0x4E75)
    a.label("setup_done")
    a.w(0x4E75)

    # d0 = raw V -> NTSC scanline. PAL returns 0; the wall clock then counts
    # whole frames only, and calibration refuses to speed-scale PAL.
    a.label("vline")
    a.abs_w(0x1239, 0xA10001)        # move.b version, d1
    a.w(0x0801)
    a.w(0x0006)                      # btst #6, d1
    a.bcc(0x66, "vline_pal")
    a.w(0x0C40)
    a.w(0x00EA)                      # cmpi.w #$EA, d0
    a.bcc(0x63, "vline_done")        # bls
    a.w(0x0440)
    a.w(0x00E5)                      # subi.w #$E5, d0
    a.w(0x0640)
    a.w(235)
    a.bcc(0x60, "vline_done")
    a.label("vline_pal")
    a.w(0x4240)                      # clr.w d0
    a.label("vline_done")
    a.w(0x4E75)

    a.label("snap")
    a.abs_w(0x33F9, VBL)
    a.l(SNAP_VBL)                    # move.w vbl, snap_vbl
    a.abs_w(0x33F9, HAND)
    a.l(SNAP_HAND)
    a.abs_w(0x3039, 0xC00008)
    a.w(0xE048)                      # lsr.w #8, d0
    a.w(0x13C0)
    a.l(SNAP_V)                      # move.b d0, snap_v
    a.w(0x4E75)

    # d0 = real scanlines since the snapshot. Long, never negative.
    a.label("wall")
    a.abs_w(0x3439, VBL)             # move.w vbl, d2
    a.abs_w(0x9479, SNAP_VBL)        # sub.w snap_vbl, d2
    a.w(0x0282)
    a.l(0x0000FFFF)                  # andi.l #$ffff, d2
    a.abs_w(0x1239, 0xA10001)
    a.w(0x0801)
    a.w(0x0006)
    a.bcc(0x66, "wall_pal")
    a.w(0xC4FC)
    a.w(262)                         # mulu.w #262, d2
    a.w(0x2F02)                      # move.l d2, -(sp)
    a.abs_w(0x3039, 0xC00008)
    a.w(0xE048)                      # lsr.w #8, d0
    a.bcc(0x61, "vline")
    a.w(0x3200)                      # move.w d0, d1
    a.w(0x1039)
    a.l(SNAP_V)                      # move.b snap_v, d0
    a.w(0x0240)
    a.w(0x00FF)
    a.bcc(0x61, "vline")
    a.w(0x9240)                      # sub.w d0, d1
    a.w(0x201F)                      # move.l (sp)+, d0
    a.w(0x48C1)                      # ext.l d1
    a.w(0xD081)                      # add.l d1, d0
    a.w(0x4A80)                      # tst.l d0
    a.bcc(0x6A, "wall_done")
    a.w(0x0680)
    a.l(262)
    a.bcc(0x60, "wall_done")
    a.label("wall_pal")
    a.w(0xC4FC)
    a.w(313)                         # mulu.w #313, d2
    a.w(0x2002)                      # move.l d2, d0
    a.label("wall_done")
    a.w(0x4E75)

    a.label("calib")
    # Bit 7 means the timing is committed. Bit 6 means the first race entry
    # has been seen and SNAP_VBL holds that vblank count. Genesis Plus GX
    # keeps the CPU at stock for 100 frames after a load, so the loop waits
    # 160 vblanks and only then counts lines. A savestate with a plain $10
    # has neither bit, so it is timed again.
    # A savestate keeps the ratio of the clock it was recorded on. A new
    # build clears that and measures again, once, after the load.
    # While the race is paused, Up+C (Start released) forgets the stored
    # ratio and runs this same wait and measure.
    a.abs_w(0x4A79, 0xFF2548)        # tst.w pause
    a.bcc(0x67, "calib_cookie")      # beq: not paused
    # The race already sampled the pad. Reading the port again here
    # returns an empty byte and the chord never matches. One player
    # leaves that sample in FF0576. Two players leave player 1 in FF0574
    # and player 2 in FF0575; FF0576 is only a copy while that rider is
    # stepped, and pause does not step.
    a.w(0x0C79)
    a.w(1)
    a.l(0xFF022C)                    # cmpi.w #1, FF022C
    a.bcc(0x66, "calib_pad1")
    a.abs_w(0x1039, 0xFF0574)        # move.b player 1, d0
    a.w(0x0200)
    a.w(0x00A1)
    a.w(0x0C00)
    a.w(0x0021)
    a.bcc(0x67, "calib_held")        # beq: player 1 already matches
    a.abs_w(0x1039, 0xFF0575)        # else try player 2
    a.bcc(0x60, "calib_chord")
    a.label("calib_pad1")
    a.abs_w(0x1039, 0xFF0576)        # move.b pad, d0
    a.label("calib_chord")
    a.w(0x0200)
    a.w(0x00A1)                      # andi.b #$A1, d0  (Start, Up, C)
    a.w(0x0C00)
    a.w(0x0021)                      # cmpi.b #$21, d0  Up+C, Start up
    a.bcc(0x66, "calib_release")
    a.label("calib_held")
    a.w(0x0839)
    a.w(0x0000)
    a.l(CALIB_HOLD)                  # btst #0, hold
    a.bcc(0x66, "calib_cookie")
    a.w(0x08F9)
    a.w(0x0000)
    a.l(CALIB_HOLD)                  # bset #0, hold
    a.abs_w(0x1039, RATIO)
    a.w(0x0800)
    a.w(0x0007)                      # btst #7, d0
    a.bcc(0x67, "calib_cookie")      # beq: a measure is already running
    a.abs_w(0x4239, RATIO)           # clr.b ratio
    a.abs_w(0x4239, CALIB_MARK)
    a.bcc(0x61, "crash_scrub")       # the stuck road offset goes with the old clock
    a.bcc(0x60, "calib_cookie")
    a.label("calib_release")
    a.w(0x08B9)
    a.w(0x0000)
    a.l(CALIB_HOLD)                  # bclr #0, hold
    a.label("calib_cookie")
    a.abs_w(0x1039, RECHECK)         # move.b recheck, d0
    a.w(0x0C00)
    a.w(0x005A)                      # cmpi.b #$5A, d0
    a.bcc(0x67, "calib_known")
    a.w(0x13FC)
    a.w(0x005A)
    a.l(RECHECK)                     # move.b #$5A, recheck
    a.abs_w(0x4239, RATIO)           # clr.b ratio
    a.label("calib_known")
    a.abs_w(0x1039, RATIO)           # move.b ratio, d0
    a.w(0x0800)
    a.w(0x0007)                      # btst #7, d0
    a.bcc(0x66, "calib_done")        # bne
    a.abs_w(0x1239, 0xA10001)
    a.w(0x0801)
    a.w(0x0006)                      # btst #6, d1
    a.bcc(0x66, "calib_pal")
    a.w(0x0800)
    a.w(0x0006)                      # btst #6, d0
    a.bcc(0x66, "calib_wait")
    a.w(0x33F9)
    a.l(VBL)
    a.l(SNAP_VBL)                    # move.w vbl, snap_vbl
    a.w(0x13FC)
    a.w(0x0040)
    a.l(RATIO)                       # move.b #$40, ratio
    a.abs_w(0x4239, CALIB_MARK)      # announcement starts when the measure lands
    a.w(0x4E75)
    a.label("calib_wait")
    a.abs_w(0x3039, VBL)             # move.w vbl, d0
    a.abs_w(0x9079, SNAP_VBL)        # sub.w snap_vbl, d0
    a.w(0x0C40)
    a.w(160)
    a.bcc(0x65, "calib_done")        # bcs: overclock may still be delayed
    # The race stores $12D7C as the vblank callback, and the pause loop calls
    # that same render with interrupts masked. A sample inside the handler
    # cannot see the next vblank. Skip that call. On the main thread, allow
    # the vblank for the two-frame count and put the mask back afterwards.
    a.w(0x0839)
    a.w(0x0000)
    a.l(IN_VBL)                      # btst #0, in_vbl
    a.bcc(0x66, "calib_done")        # bne: this call is the vblank callback
    a.w(0x40E7)                      # move.w sr, -(sp)
    a.w(0x46FC)
    a.w(0x2300)                      # move #$2300, sr
    a.bcc(0x61, "calib_sample")
    a.w(0x46DF)                      # move.w (sp)+, sr
    a.w(0x4E75)
    a.label("calib_sample")
    # Count CPU iterations across two real vblanks. The game callback is
    # skipped for these two frames, otherwise the two-player copy consumes
    # the sample and 300% is stored as the step below it. A sample that
    # never sees a vblank is discarded, and the next call tries again.
    a.w(0x50F9)
    a.l(CALIB_SPIN)                  # st spin
    a.abs_w(0x3839, VBL)             # move.w vbl, d4
    a.w(0x7400)                      # moveq #0, d2
    a.label("calib_sync")
    a.w(0x5282)                      # addq.l #1, d2
    a.w(0x0C82)
    a.l(80000)
    a.bcc(0x62, "calib_abort")       # bhi: this call never saw a vblank
    a.abs_w(0x3039, VBL)
    a.w(0xB044)                      # cmp.w d4, d0
    a.bcc(0x67, "calib_sync")
    a.abs_w(0x3839, VBL)             # move.w vbl, d4  (start of the window)
    a.w(0x7400)                      # moveq #0, d2
    a.label("calib_spin")
    a.w(0x5282)                      # addq.l #1, d2
    a.w(0x0C82)
    a.l(80000)
    a.bcc(0x62, "calib_abort")       # bhi
    a.abs_w(0x3039, VBL)
    a.w(0x9044)                      # sub.w d4, d0
    a.w(0x0C40)
    a.w(2)
    a.bcc(0x65, "calib_spin")        # bcs: still inside the two frames
    a.label("calib_counted")
    a.abs_w(0x23C2, CAL_RAW)         # move.l d2, raw count
    a.abs_w(0x4239, CALIB_SPIN)      # the copy may run again
    a.bcc(0x60, "calib_ratio")
    a.label("calib_abort")
    a.abs_w(0x4239, CALIB_SPIN)
    a.bcc(0x60, "calib_done")
    a.label("calib_ratio")
    # ratio = measured * 16 / stock. More iterations means a faster CPU.
    a.w(0x0C82)
    a.l(500)
    a.bcc(0x65, "calib_done")        # bcs: the vblank counter did not move
    a.w(0x0C82)
    a.l(40000)
    a.bcc(0x62, "calib_done")        # bhi
    a.w(0x2002)                      # move.l d2, d0
    a.w(0xE988)                      # lsl.l #4, d0
    a.w(0x80FC)
    a.w(STOCK_LINES)                 # divu.w #stock, d0
    a.w(0x0C40)
    a.w(14)
    a.bcc(0x65, "ratio_hi")          # bcs: below the stock band
    a.w(0x0C40)
    a.w(17)
    a.bcc(0x62, "ratio_hi")          # bhi
    a.w(0x303C)
    a.w(16)                          # a near-stock reading adds no delay
    a.bcc(0x60, "ratio_store")
    a.label("ratio_hi")
    a.w(0x0C40)
    a.w(255)
    a.bcc(0x63, "ratio_store")       # bls
    a.w(0x303C)
    a.w(255)
    a.label("ratio_store")
    a.w(0x0000)
    a.w(0x0080)                      # ori.b #$80, d0
    a.w(0x13C0)
    a.l(RATIO)
    a.bcc(0x60, "calib_mark")
    a.label("calib_pal")
    a.w(0x13FC)
    a.w(0x0090)
    a.l(RATIO)                       # PAL keeps the original step
    a.label("calib_mark")
    a.w(0x13FC)
    a.w(0x0001)
    a.l(CALIB_MARK)                  # move.b #1, mark
    a.bcc(0x61, "crash_scrub")
    if STOCK_LINES == 0:
        a.abs_w(0x33C2, OWED)        # d2 is the NTSC line count; PAL left it alone
    else:
        a.abs_w(0x4279, OWED)
    a.label("calib_done")
    a.w(0x4E75)
    # A crash can leave FF36E0 and the rider speed set. The road then
    # scrolls under a bike that is already still. Calibration drops both,
    # for the two human riders only, and only while the crash bit is set.
    a.label("crash_scrub")
    a.w(0x2F09)                      # move.l a1, -(sp)
    a.abs_w(0x4239, CAM_HOLD)        # a new measure must not wait on the old picture
    a.abs_w(0x42B9, 0xFF36E0)        # clr.l camera offset
    a.abs_w(0x42B9, P2_CAM)
    a.abs_w(0x4279, EST16)
    a.w(0x43F9)
    a.l(0xFF09B2)
    a.bcc(0x61, "scrub_one")
    a.w(0x43F9)
    a.l(0xFF0E32)
    a.bcc(0x61, "scrub_one")
    a.w(0x225F)                      # move.l (sp)+, a1
    a.w(0x4E75)
    a.label("scrub_one")
    a.w(0x0829)
    a.w(0x0004)
    a.w(0x0068)                      # btst #4, 104(a1)
    a.bcc(0x67, "scrub_done")
    a.w(0x4269)
    a.w(18)                          # clr.w 18(a1)
    a.w(0x42A9)
    a.w(0x0026)                      # clr.l 38(a1)
    a.w(0x08A9)
    a.w(0x0004)
    a.w(0x0068)                      # bclr #4, 104(a1)
    a.label("scrub_done")
    a.w(0x4E75)

    # The in-race pause loop renders through $12D7C and does not return to
    # the physics call, so the measure has to start from that render.
    a.label("pause_cal")
    a.abs_w(0x4A79, 0xFF2548)        # tst.w pause
    a.bcc(0x67, "pause_plain")
    a.w(0x4EB9)
    pause_jsr = len(a.buf)
    a.l(0)                           # jsr calib
    a.label("pause_plain")
    a.w(0x33FC)
    a.w(0x8134)
    a.l(0x00C00004)                  # move.w #$8134, vdp
    a.w(0x4E75)

    # While the race is paused and the camera flag is set, the loop at
    # $185EB0 skips $13D76. The measure has to run from that branch or the
    # yellow frame lasts until the pause is released.
    a.label("pause_race")
    a.w(0x4EB9)
    race_jsr = len(a.buf)
    a.l(0)                           # jsr calib
    a.w(0x4EB9)
    a.l(0x00024BDE)                  # jsr the routine this hook replaced
    a.w(0x4E75)

    # The race DMA-loads all 64 colors at the start of this vblank, and the
    # picture covers the backdrop, so color 0 alone never shows. Paint the
    # whole palette after that load. The next vblank loads the real colors
    # again before this runs, so stopping the paint restores the picture.
    # A faint yellow tint through the 160-frame wait, then one short light
    # flash per 100%.
    a.label("show_cal")
    a.abs_w(0x1039, RATIO)           # move.b ratio, d0
    a.w(0x0800)
    a.w(0x0007)                      # btst #7, d0
    a.bcc(0x66, "show_ann")
    a.w(0x0800)
    a.w(0x0006)                      # btst #6, d0
    a.bcc(0x67, "show_ret")          # beq: idle
    a.w(0x323C)
    a.w(0x0044)                      # move.w #$0044, d1  faint yellow
    a.bcc(0x61, "tint")
    a.w(0x4E75)
    a.label("show_ann")
    a.w(0x7200)                      # moveq #0, d1
    a.abs_w(0x1239, CALIB_MARK)      # move.b mark, d1
    a.bcc(0x67, "show_ret")          # beq: d0 still holds the ratio
    a.w(0x0240)
    a.w(0x007F)                      # andi.w #$7F, d0
    a.w(0x5040)                      # addq.w #8, d0  nearest 100%
    a.w(0xE848)                      # lsr.w #4, d0
    a.bcc(0x66, "show_cnt")
    a.w(0x7001)                      # moveq #1, d0
    a.label("show_cnt")
    a.w(0x0C40)
    a.w(10)
    a.bcc(0x63, "show_cap")          # bls
    a.w(0x700A)                      # moveq #10, d0
    a.label("show_cap")
    a.w(0xC0FC)
    a.w(20)                          # mulu.w #20, d0
    a.w(0xB240)                      # cmp.w d0, d1
    a.bcc(0x64, "show_fin")          # bcc: announcement is over
    a.w(0x82FC)
    a.w(20)                          # divu.w #20, d1
    a.w(0x4841)                      # swap d1  (remainder)
    a.w(0x0C41)
    a.w(4)
    a.bcc(0x65, "show_on")           # bcs: four frames lit of every twenty
    # Pause does not DMA the palette back, so a white frame would stick
    # and the blinks would only show up after unpausing.
    a.bcc(0x61, "unpaint")
    a.abs_w(0x5239, CALIB_MARK)      # addq.b #1, mark
    a.w(0x4E75)
    a.label("show_on")
    a.w(0x323C)
    a.w(0x0666)                      # move.w #$0666, d1  light flash
    a.bcc(0x61, "tint")
    a.abs_w(0x5239, CALIB_MARK)      # addq.b #1, mark
    a.w(0x4E75)
    a.label("show_fin")
    a.bcc(0x61, "unpaint")
    a.abs_w(0x4239, CALIB_MARK)
    a.w(0x08B9)
    a.w(0x0001)
    a.l(CALIB_HOLD)                  # bclr #1, hold
    a.label("show_ret")
    a.w(0x4E75)
    # The race palette with d1 ORed into every color. The picture stays
    # readable; only its brightness and hue change.
    a.label("tint")
    a.abs_w(0x2079, 0xFF1BF2)        # movea.l palette, a0
    a.w(0xB1FC)
    a.l(0)
    a.bcc(0x67, "paint")             # no palette yet: plain color
    a.w(0x23FC)
    a.l(0xC0000000)
    a.l(0x00C00004)
    a.w(0x343C)
    a.w(63)
    a.label("tint_loop")
    a.w(0x3018)                      # move.w (a0)+, d0
    a.w(0x8041)                      # or.w d1, d0
    a.abs_w(0x33C0, 0xC00000)        # move.w d0, cram
    a.dbf(2, "tint_loop")
    a.w(0x4E75)
    a.label("paint")
    a.w(0x23FC)
    a.l(0xC0000000)                  # CRAM address 0, write
    a.l(0x00C00004)
    a.w(0x343C)
    a.w(63)                          # move.w #63, d2
    a.label("paint_loop")
    a.abs_w(0x33C1, 0xC00000)        # move.w d1, cram
    a.dbf(2, "paint_loop")
    a.w(0x4E75)
    # Copy the race palette the game DMA'd from. Writing CRAM directly left
    # the last white frame in place for the whole pause.
    a.label("unpaint")
    a.abs_w(0x2079, 0xFF1BF2)        # movea.l palette, a0
    a.w(0xB1FC)
    a.l(0)                           # cmpa.l #0, a0
    a.bcc(0x67, "show_ret")          # beq
    a.w(0x23FC)
    a.l(0xC0000000)
    a.l(0x00C00004)
    a.w(0x343C)
    a.w(63)                          # move.w #63, d2
    a.label("unpaint_loop")
    a.w(0x33D8)
    a.l(0x00C00000)                  # move.w (a0)+, cram
    a.dbf(2, "unpaint_loop")
    a.w(0x4E75)

    if STOCK_LINES:
        # OWED arrives as the previous step's stock cost. Leave 1..16 there:
        # 16 means this frame lasted at least as long as that cost.
        a.label("scale_now")
        a.abs_w(0x3239, OWED)        # move.w owed, d1
        a.bcc(0x67, "scale_full")    # beq
        a.w(0x3F01)                  # move.w d1, -(sp)
        a.w(0x4EB9)
        wall_abs = len(a.buf)
        a.l(0)                       # jsr wall, d0 = lines since the last step
        a.w(0x7400)                  # moveq #0, d2
        a.w(0x341F)                  # move.w (sp)+, d2
        a.w(0xB082)                  # cmp.l d2, d0
        a.bcc(0x64, "scale_full")    # bcc: this frame was not shorter than stock
        a.w(0xE988)                  # lsl.l #4, d0
        a.w(0x80C2)                  # divu.w d2, d0
        a.w(0x4A40)                  # tst.w d0
        a.bcc(0x66, "scale_store")   # bne
        a.w(0x7001)                  # moveq #1, d0
        a.bcc(0x60, "scale_store")
        a.label("scale_full")
        a.w(0x7010)                  # moveq #16, d0
        a.label("scale_store")
        a.abs_w(0x33C0, OWED)        # move.w d0, owed
        a.w(0x4E75)

        # d0.l signed *= owed/16. 0 and 16+ leave d0 alone (full vanilla step).
        a.label("scale_delta")
        a.w(0x2F01)                  # move.l d1, -(sp)
        a.abs_w(0x3239, OWED)        # move.w owed, d1
        a.bcc(0x67, "delta_pop")
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x64, "delta_pop")     # bcc: full step, d0 stays
        a.w(0x2F02)                  # move.l d2, -(sp)
        a.w(0x3401)                  # move.w d1, d2
        a.w(0x4A80)                  # tst.l d0
        a.bcc(0x6A, "delta_pos")
        a.w(0x4480)                  # neg.l d0
        a.w(0x3F3C)
        a.w(1)
        a.bcc(0x60, "delta_mul")
        a.label("delta_pos")
        a.w(0x4267)                  # clr.w -(sp)
        a.label("delta_mul")
        a.w(0x2200)                  # move.l d0, d1
        a.w(0x4841)                  # swap d1
        a.w(0xC2C2)                  # mulu.w d2, d1
        a.w(0x4841)
        a.w(0x4241)                  # clr.w d1
        a.w(0xC0C2)                  # mulu.w d2, d0
        a.w(0xD081)                  # add.l d1, d0
        a.w(0xE888)                  # lsr.l #4, d0
        a.w(0x4A5F)                  # tst.w (sp)+
        a.bcc(0x67, "delta_nosign")
        a.w(0x4480)                  # neg.l d0
        a.label("delta_nosign")
        a.w(0x241F)                  # move.l (sp)+, d2
        a.label("delta_pop")
        a.w(0x221F)                  # move.l (sp)+, d1
        a.w(0x4E75)
        # Crash camera. Bit 4 keeps FF36E0 and either adds the frame
        # step or pulls the offset back by a quarter. The bike itself has
        # already stopped updating, so the road keeps sliding under it
        # until that offset is gone. A straight throw never stores one.
        # On an overclock, a stopped bike drops the offset now.
        a.label("cam_idle")
        a.w(0x7200)                  # moveq #0, d1
        a.abs_w(0x1239, RATIO)       # move.b ratio, d1
        a.w(0x0801)
        a.w(0x0007)                  # btst #7, d1
        a.bcc(0x67, "cam_go")
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "cam_go")        # beq: stock camera
        a.w(0x4A69)
        a.w(18)                      # tst.w 18(a1) speed
        a.bcc(0x66, "cam_go")
        a.abs_w(0x42B9, 0xFF36E0)    # clr.l skid
        a.w(0x7000)                  # moveq #0, d0
        a.w(0x7201)                  # moveq #1, d1
        a.w(0x4E75)
        a.label("cam_go")
        a.w(0x7200)                  # moveq #0, d1
        a.w(0x4E75)
        # 38(a1) = speed * frame length, then the bike advances by it.
        # After a crash, bit 4 stays set until the riding state returns,
        # and that state returns only once speed hits zero. Friction is
        # not applied in between, so a sideways hit leaves the speed in
        # and the road keeps scrolling under a rider who is already back
        # on the bike. A straight hit often does not. On an overclock,
        # a human who is on the wheels again and is not holding gas drops
        # that speed. Running back (bit 3 of 105) still uses it.
        a.label("coast_stop")
        a.w(0x7200)
        a.abs_w(0x1239, RATIO)
        a.w(0x0801)
        a.w(0x0007)
        a.bcc(0x67, "coast_mul")
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "coast_mul")
        a.w(0x0829)
        a.w(0x0004)
        a.w(0x0068)                  # btst #4, 104(a1)
        a.bcc(0x67, "coast_mul")
        a.w(0x0829)
        a.w(0x0003)
        a.w(0x0069)                  # btst #3, 105(a1)
        a.bcc(0x66, "coast_mul")     # bne: still off the bike
        a.w(0xB3FC)
        a.l(0xFF0E32)
        a.bcc(0x67, "coast_p2")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x66, "coast_mul")     # opponent keeps the original coast
        a.w(0x0C79)
        a.w(1)
        a.l(0xFF022C)
        a.bcc(0x66, "coast_one")
        a.abs_w(0x1239, 0xFF0574)    # 2P player 1
        a.bcc(0x60, "coast_gas")
        a.label("coast_one")
        a.abs_w(0x1239, 0xFF0576)
        a.bcc(0x60, "coast_gas")
        a.label("coast_p2")
        a.abs_w(0x1239, 0xFF0575)
        a.label("coast_gas")
        a.w(0x0801)
        a.w(0x0004)                  # btst #4, d1  gas
        a.bcc(0x66, "coast_mul")
        a.w(0x4240)                  # clr.w d0
        a.w(0x4269)
        a.w(18)                      # clr.w 18(a1)
        a.label("coast_mul")
        a.abs_w(0x3439, 0xFF1BDC)    # move.w frame, d2
        a.w(0xC0C2)                  # mulu.w d2, d0
        a.w(0x2340)
        a.w(0x0026)                  # move.l d0, 38(a1)
        a.w(0x4E75)

        a.label("scale_sub")
        a.bcc(0x61, "cam_idle")
        a.w(0x4A41)                  # tst.w d1
        a.bcc(0x66, "scale_sub_done")
        a.bcc(0x61, "scale_delta")
        a.w(0x91B9)
        a.l(0xFF36E0)                # sub.l d0, skid
        a.label("scale_sub_done")
        a.w(0x4E75)
        a.label("scale_add")
        a.bcc(0x61, "cam_idle")
        a.w(0x4A41)                  # tst.w d1
        a.bcc(0x66, "scale_add_done")
        a.bcc(0x61, "scale_delta")
        a.w(0xD1B9)
        a.l(0xFF36E0)                # add.l d0, skid
        a.label("scale_add_done")
        a.w(0x4E75)

        # Player 2's camera call. Keep his offset out of the shared word
        # while his view is built, then put player 1's word back.
        # $13088 is the full upload's "picture taken". The partial tail
        # jumps to $12FD2 and never gets here, so a short chunk keeps the hold.
        a.label("cam_release")
        a.abs_w(0x4239, CAM_HOLD)    # clr.b hold
        a.abs_w(0x2479, 0xFF278E)    # movea.l FF278E, a2
        a.w(0x4E75)
        # $131D6 publishes the picture. The hold used to be set after this
        # returned, so the upload could take the picture and clear the hold
        # before it was set. The next step then waited for an upload that
        # had already happened, and the frame length jumped. Set the hold
        # and the flag together, with the vblank masked.
        a.label("arm_pic")
        a.w(0x40E7)                  # move.w sr, -(sp)
        a.w(0x007C)
        a.w(0x0700)                  # ori.w #$0700, sr
        a.w(0x0C39)
        a.w(0x0090)
        a.l(RATIO)
        a.bcc(0x67, "arm_flag")
        a.w(0x0839)
        a.w(0x0007)
        a.l(RATIO)
        a.bcc(0x67, "arm_flag")
        a.w(0x0C79)
        a.w(1)
        a.l(0xFF022A)
        a.bcc(0x66, "arm_flag")
        a.w(0x50F9)
        a.l(CAM_HOLD)
        a.label("arm_flag")
        a.abs_w(0x2479, 0xFF278A)    # movea.l FF278A, a2
        a.w(0x34BC)
        a.w(1)                       # move.w #1, (a2)
        a.w(0x46DF)                  # move.w (sp)+, sr
        a.w(0x4E75)
        a.label("cam_p2")
        a.w(0x2F39)
        a.l(0xFF36E0)                # move.l skid, -(sp)
        a.w(0x23F9)
        a.l(P2_CAM)
        a.l(0xFF36E0)                # move.l p2_cam, skid
        a.w(0x4EB9)
        a.l(0x182BC)                 # jsr camera
        a.w(0x23F9)
        a.l(0xFF36E0)
        a.l(P2_CAM)                  # move.l skid, p2_cam
        a.w(0x23DF)
        a.l(0xFF36E0)                # move.l (sp)+, skid
        a.w(0x4E75)

        # His 1P digits sit in the window buffer. A partial upload never
        # sends that buffer, so write the same two cells straight to VRAM.
        # a2 is the VDP control port and a1 the data port, as in his frame.
        a.label("fps_full")
        a.bcc(0x61, "fps_poke")
        a.w(0x4EF9)
        a.l(0x12FA4)
        a.label("fps_part")
        a.bcc(0x61, "fps_poke")
        a.w(0x4EF9)
        a.l(0x12FD2)
        a.label("fps_poke")
        a.abs_w(0x4A79, 0xFF022A)
        a.bcc(0x66, "fps_ret")
        a.abs_w(0x4A79, 0xFF753E)    # SHOW FPS
        a.bcc(0x67, "fps_ret")
        a.w(0x24BC)
        a.l(0x7D1A0000)              # VRAM $3D1A
        a.w(0x22B9)
        a.l(0xFFEAC8)                # move.l digits, (a1)
        a.label("fps_ret")
        a.w(0x4E75)

        a.label("charge")
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)                  # btst #7, d0
        a.bcc(0x67, "charge_bail")   # beq: timing not committed yet
        a.bcc(0x60, "charge_run")
        a.label("charge_bail")
        a.w(0x4E75)
        a.label("charge_run")
        a.w(0x4EB9)
        wall_chg = len(a.buf)
        a.l(0)
        a.w(0x2400)                  # move.l d0, d2
        a.abs_w(0x3039, HAND)
        a.abs_w(0x9079, SNAP_HAND)   # sub.w snap_hand, d0
        a.w(0x0280)
        a.l(0x0000FFFF)
        a.w(0x9480)                  # sub.l d0, d2
        a.bcc(0x6A, "main_pos")
        a.w(0x7400)                  # moveq #0, d2
        a.label("main_pos")
        a.w(0x7200)
        a.w(0x1239)
        a.l(RATIO)                   # move.b ratio, d1
        a.w(0x0241)
        a.w(0x007F)                  # andi.w #$7F, d1  (drop the committed bit)
        a.w(0x2002)                  # move.l d2, d0
        a.w(0xC0C1)                  # mulu.w d1, d0
        a.w(0xE888)                  # lsr.l #4, d0
        a.w(0x33C0)
        a.l(OWED)
        a.w(0x4E75)

        # $1E47A and $1E418 load the step path (speed * vblanks) >> 8. The
        # heading multiplies yaw by it and the sideways step multiplies by it
        # again, so a short overclocked frame loses turn and the slide
        # threshold. Load the path of a 100% frame instead.
        a.label("scale_hdg")
        a.abs_w(0x4A79, FRAME16)     # tst.w frame16
        a.bcc(0x67, "hdg_plain")
        a.w(0x7000)                  # moveq #0, d0
        a.w(0x3029)
        a.w(0x0012)                  # move.w 18(a1), d0
        a.abs_w(0xC0F9, FRAME16)     # mulu.w frame16, d0
        a.w(0xE888)                  # lsr.l #4, d0
        a.w(0xE088)                  # lsr.l #8, d0
        a.w(0x4E75)
        a.label("hdg_plain")
        a.w(0x2029)
        a.w(0x0026)                  # move.l 38(a1), d0
        a.w(0xE088)                  # lsr.l #8, d0
        a.w(0x4E75)

        # d0 is the sideways step of a 100% frame. 34(a1) keeps that for the
        # lean and the rider contacts. The position takes dt/stock of it,
        # and the remainder waits in this rider's word of LAT_REM.
        a.label("lat_step")
        a.w(0x3340)
        a.w(0x0022)                  # move.w d0, 34(a1)
        a.abs_w(0x4A79, FRAME16)
        a.bcc(0x67, "lat_add")
        a.w(0x48E7)
        a.w(0x6080)                  # movem.l d1-d2/a0, -(sp)
        a.w(0x41F9)
        a.l(LAT_REM + 32)            # lea spare word, a0
        a.w(0x2409)                  # move.l a1, d2
        a.w(0x0482)
        a.l(0xFF09B2)                # subi.l #first rider, d2
        a.bcc(0x65, "lat_go")        # bcs
        a.w(0x0C82)
        a.l(0x800)
        a.bcc(0x64, "lat_go")        # bcc: past the last rider
        a.w(0xEC8A)                  # lsr.l #6, d2  slot * 2
        a.w(0x41F9)
        a.l(LAT_REM)
        a.w(0xD0C2)                  # adda.w d2, a0
        a.label("lat_go")
        a.abs_w(0xC1F9, 0xFF1BDC)    # muls.w frame, d0
        a.w(0xE980)                  # asl.l #4, d0
        a.w(0x3210)                  # move.w (a0), d1
        a.w(0x48C1)                  # ext.l d1
        a.w(0xD081)                  # add.l d1, d0
        a.abs_w(0x81F9, FRAME16)     # divs.w frame16, d0
        a.w(0x4840)                  # swap d0
        a.w(0x3080)                  # move.w d0, (a0)  remainder
        a.w(0x4840)
        a.w(0x4CDF)
        a.w(0x0106)                  # movem.l (sp)+, d1-d2/a0
        a.label("lat_add")
        a.w(0xD151)                  # add.w d0, (a1)
        a.w(0x4E75)

        # $18836 and $19262 scroll the mountains by the player's sideways
        # step. 34(a1) holds the step of a 100% frame, so take the part this
        # frame moved. Each site keeps its own remainder.
        for i, rem in enumerate((HILL_REM, HILL_REM + 2)):
            a.label(f"hill_lat{i}")
            a.abs_w(0x3239, 0xFF09D4)    # move.w step, d1
            a.abs_w(0x4A79, FRAME16)
            a.bcc(0x67, f"hill_out{i}")
            a.w(0x2F00)                  # move.l d0, -(sp)
            a.w(0x3001)                  # move.w d1, d0
            a.abs_w(0xC1F9, 0xFF1BDC)    # muls.w frame, d0
            a.w(0xE980)                  # asl.l #4, d0
            a.abs_w(0x3239, rem)
            a.w(0x48C1)                  # ext.l d1
            a.w(0xD081)                  # add.l d1, d0
            a.abs_w(0x81F9, FRAME16)     # divs.w frame16, d0
            a.w(0x3200)                  # move.w d0, d1
            a.w(0x4840)
            a.abs_w(0x33C0, rem)         # remainder
            a.w(0x201F)
            a.label(f"hill_out{i}")
            a.w(0x4A41)                  # tst.w d1
            a.w(0x4E75)

        # The road bend adds (speed * vblanks >> 8) * curve change to 54(a1),
        # which the sideways step subtracts from a 100% heading. Build it
        # with the speed of a 100% frame too.
        a.label("scale_lane")
        a.w(0x2F00)                  # move.l d0, -(sp)
        a.w(0x2229)
        a.w(0x0026)                  # move.l 38(a1), d1
        a.abs_w(0x3039, FRAME16)     # move.w frame16, d0
        a.bcc(0x67, "lane_plain")
        a.w(0x7200)                  # moveq #0, d1
        a.w(0x3229)
        a.w(0x0012)                  # move.w 18(a1), d1
        a.w(0xC2C0)                  # mulu.w d0, d1  speed of a 100% frame
        a.w(0xE889)                  # lsr.l #4, d1
        a.label("lane_plain")
        a.w(0xE089)                  # lsr.l #8, d1
        a.w(0xC3C6)                  # muls.w d6, d1
        a.w(0xE081)
        a.w(0xC3FC)
        a.w(0x0300)
        a.w(0xE081)
        a.w(0x201F)                  # move.l (sp)+, d0
        a.w(0x4E75)

        # A 100% frame is this one times the clock ratio, averaged over
        # about four steps. FRAME16 is that length in 1/16 vblank when this
        # frame is shorter, else 0 and the original step runs.
        a.label("est_update")
        a.w(0x48E7)
        a.w(0xE000)                  # movem.l d0-d2, -(sp)
        a.w(0x7200)
        a.abs_w(0x1239, RATIO)
        a.bcc(0x6A, "est_off")       # bpl: not committed
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "est_off")
        a.w(0x7000)
        a.abs_w(0x3039, 0xFF1BDC)    # move.w frame, d0
        a.w(0xC0C1)                  # mulu.w d1, d0
        a.w(0x0C80)
        a.l(160)
        a.bcc(0x63, "est_cap")
        a.w(0x203C)
        a.l(160)
        a.label("est_cap")
        a.abs_w(0x3239, EST16)
        a.bcc(0x66, "est_blend")
        a.w(0x3200)                  # first sample
        a.bcc(0x60, "est_have")
        a.label("est_blend")
        a.w(0x3401)
        a.w(0xD241)
        a.w(0xD242)                  # d1 = 3 * avg
        a.w(0xD240)
        a.w(0xE449)                  # lsr.w #2, d1
        a.label("est_have")
        a.abs_w(0x33C1, EST16)
        if cd:
            # His FPS cap sets his overclocked frame, not the CPU, so
            # frame * ratio does not give his solo 100% frame. Steering
            # grows with the frame, so solo uses a fixed length instead.
            a.w(0x0C79)
            a.w(1)
            a.l(0xFF022C)
            a.bcc(0x67, "est_two")
            a.w(0x323C)
            a.w(TI_SOLO_FRAME16)     # move.w #frame, d1
            a.label("est_two")
        a.w(0x0C41)
        a.w(32)
        a.bcc(0x64, "est_lo")
        a.w(0x7220)                  # moveq #32, d1
        a.label("est_lo")
        a.w(0x0C41)
        a.w(160)
        a.bcc(0x63, "est_hi")
        a.w(0x323C)
        a.w(160)
        a.label("est_hi")
        a.w(0x7000)
        a.abs_w(0x3039, 0xFF1BDC)
        a.w(0xE948)                  # lsl.w #4, d0
        a.w(0xB041)                  # cmp.w d1, d0
        a.bcc(0x64, "est_off")       # bcc: a 100% frame would not be longer
        a.abs_w(0x33C1, FRAME16)
        a.bcc(0x60, "est_out")
        a.label("est_off")
        a.abs_w(0x4279, FRAME16)
        a.label("est_out")
        a.w(0x4CDF)
        a.w(0x0007)
        a.w(0x4E75)


        # $1E712 adds the whole hill step on the frame a segment boundary is
        # crossed, then $1E894 copies that into the drawn height. At 300% the
        # frames between crossings draw nothing, so a rise or a jump pops.
        # Spread the segment's height across the distance actually traveled.
        # The lane step is left alone. 100% still executes the original add.
        # 126(a1) is the unused last word of a rider slot and holds the
        # fractional unit so the pieces add back up to the original step.
        a.label("height_add")
        a.w(0x2F02)                  # move.l d2, -(sp)
        a.w(0x7000)                  # moveq #0, d0
        a.abs_w(0x1039, RATIO)       # move.b ratio, d0
        a.w(0x0800)
        a.w(0x0007)                  # btst #7, d0
        a.bcc(0x67, "height_plain")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "height_plain")
        a.w(0xB3FC)
        a.l(0xFF09B2)                # cmpa.l #first rider, a1
        a.bcc(0x65, "height_plain")  # blo
        a.w(0xB3FC)
        a.l(0xFF11B2)                # cmpa.l #past last rider, a1
        a.bcc(0x64, "height_plain")  # bhs
        a.w(0x2429)
        a.w(0x0026)                  # move.l 38(a1), d2  distance
        a.w(0x4842)                  # swap d2
        a.w(0x4A42)                  # tst.w d2
        a.bcc(0x66, "height_plain")  # bne: more than one segment, keep the lump
        a.w(0x4842)                  # swap d2
        a.w(0x3029)
        a.w(0x0006)                  # move.w 6(a1), d0  position inside the segment
        a.w(0xB042)                  # cmp.w d2, d0
        a.bcc(0x65, "height_cross")  # bcs: this frame finished a segment
        a.w(0x102C)
        a.w(0x0001)                  # move.b 1(a4), d0  slope of the current segment
        a.w(0x4880)                  # ext.w d0
        a.w(0x3202)                  # move.w d2, d1
        a.bcc(0x61, "hmul")
        a.bcc(0x60, "height_out")
        a.label("height_cross")
        a.w(0x3202)                  # move.w d2, d1
        a.w(0x9240)                  # sub.w d0, d1  distance still on the old slope
        a.w(0x3F00)                  # move.w d0, -(sp)  distance already on the new one
        a.w(0x3007)                  # move.w d7, d0  old slope
        a.bcc(0x61, "hmul")
        a.w(0x321F)                  # move.w (sp)+, d1
        a.w(0x102C)
        a.w(0x0001)                  # move.b 1(a4), d0
        a.w(0x4880)                  # ext.w d0
        a.bcc(0x61, "hmul")
        a.bcc(0x60, "height_out")
        a.label("height_plain")
        a.w(0xDF69)
        a.w(0x003A)                  # add.w d7, 58(a1)
        a.label("height_out")
        a.w(0x241F)                  # move.l (sp)+, d2
        a.w(0x3029)
        a.w(0x0006)                  # move.w 6(a1), d0
        a.w(0x4E75)
        # d0 = signed slope, d1 = distance. Fold both into 58(a1).
        a.label("hmul")
        a.w(0x3401)                  # move.w d1, d2
        a.w(0xE24A)                  # lsr.w #1, d2
        a.w(0xC5C0)                  # muls.w d0, d2
        a.w(0xD482)                  # add.l d2, d2
        a.w(0x0801)
        a.w(0x0000)                  # btst #0, d1
        a.bcc(0x67, "hmul_even")
        a.w(0x3200)                  # move.w d0, d1
        a.w(0x48C1)                  # ext.l d1
        a.w(0xD481)                  # add.l d1, d2
        a.label("hmul_even")
        a.w(0x7200)                  # moveq #0, d1
        a.w(0x3229)
        a.w(0x007E)                  # move.w 126(a1), d1
        a.w(0xD481)                  # add.l d1, d2
        a.w(0x3342)
        a.w(0x007E)                  # move.w d2, 126(a1)
        a.w(0x4842)                  # swap d2
        a.w(0xD569)
        a.w(0x003A)                  # add.w d2, 58(a1)
        a.w(0x4E75)

        # $1E778 already adds (slope * position in the segment) on top of
        # 58(a1). The spread above puts that same partial into 58(a1), so the
        # bike climbs and then drops by a whole segment every crossing. When
        # the spread is active, draw 58(a1) alone. 60(a1) still keeps the
        # original partial. A multi-segment frame keeps the original sum.
        a.label("height_join")
        a.w(0x3341)
        a.w(0x003C)                  # move.w d1, 60(a1)
        a.w(0x7000)                  # moveq #0, d0
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)                  # btst #7, d0
        a.bcc(0x67, "join_keep")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "join_keep")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x65, "join_keep")     # blo
        a.w(0xB3FC)
        a.l(0xFF11B2)
        a.bcc(0x64, "join_keep")     # bhs
        a.w(0x2029)
        a.w(0x0026)                  # move.l 38(a1), d0
        a.w(0x4840)                  # swap d0
        a.w(0x4A40)                  # tst.w d0
        a.bcc(0x66, "join_keep")     # bne: more than one segment
        a.w(0x7200)                  # moveq #0, d1
        a.label("join_keep")
        a.w(0xD269)
        a.w(0x003A)                  # add.w 58(a1), d1
        a.w(0x4E75)

        # $1E7F8. FF098A is cleared every call and rewritten only when the
        # slope increases, to the height ahead of the bike. Using that as
        # the launch base on overclock lifts the bike off the road: the
        # shadow comes back and the climb steers as if the wheels are light.
        # 58(a1) is the height already spread along the slope. Stock still
        # reads FF098A, because at 100% that word matches the original curve.
        a.label("curve_base")
        a.w(0x7000)                  # moveq #0, d0
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)
        a.bcc(0x67, "curve_stock")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "curve_stock")
        a.w(0x3029)
        a.w(0x003A)                  # move.w 58(a1), d0
        a.w(0x4E75)
        a.label("curve_stock")
        a.abs_w(0x3039, 0xFF098A)    # move.w FF098A, d0
        a.w(0x4E75)

        # $1E82C. Distance/speed on a 2-vblank frame is often 0, and the
        # game then forces one full frame of flight. That lift clears the
        # road while it is still rising, so the bike is airborne for a hop:
        # the lateral step is skipped, heading keeps integrating, and a held
        # turn falls off. If the road has started to fall, a kick saved on
        # the previous climb frame is larger than the new slope's kick and
        # replaces it, so the crest still leaves the ground. Stock is the
        # original store and divide. d5 is restored by the caller.
        a.label("kick_keep")
        a.w(0x3340)
        a.w(0x002C)                  # move.w d0, 44(a1)
        a.w(0x7000)
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)
        a.bcc(0x67, "kick_div")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "kick_div")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x66, "kick_div")      # bne: only the player carries a kick
        a.abs_w(0x3039, KICK)        # move.w kick, d0
        a.bcc(0x67, "kick_div")
        a.abs_w(0x4279, KICK)        # clr.w kick
        a.w(0x3A29)
        a.w(0x003A)                  # move.w 58(a1), d5
        a.w(0x9A69)
        a.w(0x0002)                  # sub.w 2(a1), d5
        a.bcc(0x6A, "kick_div")      # bpl: road has not fallen
        a.w(0xB069)
        a.w(0x002C)                  # cmp.w 44(a1), d0
        a.bcc(0x6F, "kick_div")      # ble
        a.w(0x3340)
        a.w(0x002C)                  # move.w d0, 44(a1)
        a.label("kick_div")
        a.w(0x88C3)                  # divu.w d3, d4
        a.w(0x4E75)

        # $1E872. Above, with the road still rising or flat, draw the road
        # and do not set the airborne bit. 44(a1) stays, and the player's
        # kick is copied aside for the crest. A falling road takes the air
        # path. The original compare is kept for stock and for a bike that
        # is already flying.
        a.label("stick")
        a.w(0x0829)
        a.w(0x0000)
        a.w(0x0068)                  # btst #0, 104(a1)
        a.bcc(0x66, "stick_cmp")
        a.w(0x7000)
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)
        a.bcc(0x67, "stick_cmp")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "stick_cmp")
        a.w(0x3002)                  # move.w d2, d0
        a.w(0x9041)                  # sub.w d1, d0
        a.bcc(0x6F, "stick_down")    # ble: curve is not above the road
        a.w(0x3629)
        a.w(0x003A)                  # move.w 58(a1), d3
        a.w(0x9669)
        a.w(0x0002)                  # sub.w 2(a1), d3
        a.bcc(0x6B, "stick_fly")     # bmi: the road fell
        a.w(0x3001)                  # move.w d1, d0
        a.bcc(0x61, "damp")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x66, "stick_hold")
        a.w(0x3029)
        a.w(0x002C)                  # move.w 44(a1), d0
        a.abs_w(0x33C0, KICK)        # move.w d0, kick
        a.label("stick_hold")
        a.w(0x4269)
        a.w(0x002E)                  # clr.w 46(a1)
        a.w(0x4EF9)
        a.l(0x001E8CA)               # return, leaving 44(a1) intact
        a.label("stick_cmp")
        a.w(0xB242)                  # cmp.w d2, d1
        a.bcc(0x6A, "stick_down")
        a.label("stick_fly")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x66, "stick_air")
        a.abs_w(0x4279, KICK)
        a.label("stick_air")
        a.w(0x4EF9)
        a.l(0x001E878)
        a.label("stick_down")
        a.w(0xB3FC)
        a.l(0xFF09B2)
        a.bcc(0x66, "stick_go")
        a.abs_w(0x4279, KICK)
        a.label("stick_go")
        a.w(0x4EF9)
        a.l(0x001E894)

        # $1A3C0 and the copy at $1A5C4. The climbed fraction of this
        # segment's slope is a sawtooth on every strip. Holding the rows
        # at one screen height snaps the near edge when the segment
        # changes. The foreground stays put and the unclimbed remainder
        # goes to the next row, which meets the row after it. The rider's
        # raw height gets that fraction back once, so the bike does not
        # bob. Stock still negates into d3. Returns past the vanilla
        # block (return address + $1C). d0 and d2 are free; d5 is only read.
        a.label("road_slope")
        a.w(0x4883)                  # ext.w d3
        a.w(0x3003)                  # move.w d3, d0  full slope
        a.bcc(0x67, "rs_part")
        a.bcc(0x6B, "rs_neg")
        a.w(0xC6C5)                  # mulu.w d5, d3
        a.w(0x4843)                  # swap d3
        a.bcc(0x60, "rs_part")
        a.label("rs_neg")
        a.w(0x4443)                  # neg.w d3
        a.w(0xC6C5)
        a.w(0x4843)
        a.w(0x4443)                  # neg.w d3  signed partial
        a.label("rs_part")
        a.abs_w(0x33C3, 0xFF092C)    # move.w d3, partial
        a.w(0x3400)                  # move.w d0, d2
        a.w(0x9443)                  # sub.w d3, d2  remaining = full - partial
        a.w(0x4443)                  # neg.w d3     vanilla foreground = -partial
        a.w(0x7200)                  # moveq #0, d1
        a.abs_w(0x1239, RATIO)       # move.b ratio, d1
        a.w(0x0801)
        a.w(0x0007)
        a.bcc(0x67, "rs_exit")
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "rs_exit")
        a.abs_w(0x33C2, ROAD_STEP)   # remainder goes to the next row
        a.abs_w(0x3239, 0xFF092C)    # move.w partial, d1
        a.abs_w(0x33C1, ROAD_FIX)    # rider raw gets it once
        a.w(0x7600)                  # moveq #0, d3  foreground stays put
        a.label("rs_exit")
        a.w(0x2017)                  # move.l (sp), d0
        a.w(0x0680)
        a.l(0x1C)                    # addi.l #$1C, d0
        a.w(0x2E80)                  # move.l d0, (sp)
        a.w(0x4E75)

        # Bottom of the strip loop. Stock adds the slope just read.
        # Overclock adds the saved step, keeps this slope for the next
        # row, and on the first row only puts the climbed fraction back
        # into the raw height the rider uses. The polygon already used d3.
        a.label("slope_add")
        a.w(0x4882)                  # ext.w d2
        a.w(0x7200)
        a.abs_w(0x1239, RATIO)
        a.w(0x0801)
        a.w(0x0007)
        a.bcc(0x67, "sa_stock")
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "sa_stock")
        a.w(0x3228)
        a.w(0xFFFE)                  # move.w -2(a0), d1
        a.abs_w(0xD279, ROAD_FIX)    # add.w fix, d1
        a.w(0x3141)
        a.w(0xFFFE)                  # move.w d1, -2(a0)
        a.abs_w(0x4279, ROAD_FIX)    # clr.w fix  (once)
        a.abs_w(0x3239, ROAD_STEP)   # move.w step, d1
        a.w(0xD641)                  # add.w d1, d3
        a.abs_w(0x33C2, ROAD_STEP)   # move.w d2, step
        a.w(0x4E75)
        a.label("sa_stock")
        a.w(0xD642)                  # add.w d2, d3
        a.w(0x4E75)
        # The loop head is more than 32 KB away, so dbf stays local and
        # an absolute jump returns to the strip. Falling off the dbf
        # returns to the instruction after the hook.
        a.label("slope_step_a")
        a.bcc(0x61, "slope_add")
        a.dbf(0, "sa_more_a")
        a.w(0x4E75)
        a.label("sa_more_a")
        a.w(0x4EF9)
        a.l(0x1A408)
        a.label("slope_step_b")
        a.bcc(0x61, "slope_add")
        a.dbf(0, "sa_more_b")
        a.w(0x4E75)
        a.label("sa_more_b")
        a.w(0x4EF9)
        a.l(0x1A5F8)

        # Two players upload one picture across several vblanks (FF3236).
        # The DMA itself is the same size as the one-player upload and does
        # not get faster when the CPU does. A fast CPU finishes the copy
        # early enough that the whole upload fits in this vblank. A milder
        # overclock does not: the copy takes the first vblank, the DMA the
        # next, so the top of the screen is not overwritten. 100% keeps
        # the original table. 3.0x and above is the one-vblank path.
        # Entered by jmp in place of that table; every exit is a jump.
        a.label("two_dma")
        a.w(0x7200)
        a.abs_w(0x1239, RATIO)
        a.w(0x0801)
        a.w(0x0007)
        a.bcc(0x67, "two_stock")
        a.w(0x0241)
        a.w(0x007F)
        a.w(0x0C41)
        a.w(16)
        a.bcc(0x67, "two_stock")
        a.w(0x0C41)
        a.w(48)
        a.bcc(0x64, "two_fast")      # bcc: 3.0x and above
        a.w(0x0C79)
        a.w(1)
        a.l(0xFF3236)
        a.bcc(0x67, "two_rows")      # second vblank is the DMA
        a.w(0x33FC)
        a.w(1)
        a.l(0xFF3236)
        a.w(0x4EB9)
        a.l(0x13CE2)
        a.w(0x0C79)
        a.w(1)
        a.l(0xFF7A32)
        a.bcc(0x66, "two_half")
        a.abs_w(0x4279, 0xFF7A32)
        a.w(0x203C)
        a.l(0xFA00)
        a.w(0x41F9)
        a.l(0xFF59A6)
        a.w(0x4EB9)
        a.l(0x11BA8)
        a.label("two_half")
        a.w(0x4EF9)
        a.l(0x12FD2)
        a.label("two_fast")
        a.w(0x4EB9)
        a.l(0x13CE2)
        a.w(0x0C79)
        a.w(1)
        a.l(0xFF7A32)                # cmpi.w #1, FF7A32
        a.bcc(0x66, "two_rows")
        a.abs_w(0x4279, 0xFF7A32)
        a.w(0x203C)
        a.l(0xFA00)
        a.w(0x41F9)
        a.l(0xFF59A6)
        a.w(0x4EB9)
        a.l(0x11BA8)
        a.label("two_rows")
        a.abs_w(0x2039, 0xFF27A2)
        a.w(0x323C)
        a.w(0x3000)
        a.w(0x343C)
        a.w(0x0400)
        a.w(0x4EB9)
        a.l(0x17996)
        a.abs_w(0x2039, 0xFF27A2)
        a.w(0x0640)
        a.w(0x0800)
        a.w(0x323C)
        a.w(0x3800)
        a.w(0x343C)
        a.w(0x0340)
        a.w(0x4EB9)
        a.l(0x17996)
        a.abs_w(0x4A79, 0xFF2546)
        a.bcc(0x6B, "two_spr")
        a.abs_w(0x2039, 0xFF1BF2)
        a.w(0x323C)
        a.w(0)
        a.w(0x343C)
        a.w(0x0040)
        a.w(0x4EB9)
        a.l(0x17A6A)
        a.label("two_spr")
        a.abs_w(0x2039, 0xFF2776)
        a.w(0x323C)
        a.w(0xC000)
        a.w(0x343C)
        a.w(0x0340)
        a.w(0x4EB9)
        a.l(0x17996)
        a.w(0x4EB9)
        a.l(0x13088)
        a.w(0x4EB9)
        a.l(0x12C5E)
        a.abs_w(0x4279, 0xFF3236)
        a.w(0x4EF9)
        a.l(0x12FD2)
        a.label("two_stock")
        a.abs_w(0x3039, 0xFF3236)
        a.w(0xE540)
        a.w(0x41F9)
        a.l(0x12E5E)
        a.w(0x4EF0)
        a.w(0)

        # Restart is a soft reset: $200 sees $A1000C and skips the RAM
        # clear, then jumps here. A one-player race left in RAM is then
        # still there when two players start, and the second view scrolls
        # with the first. Wipe work RAM. A committed ratio and its cookie
        # stay, so the clock is not measured again. The reset stack lives
        # at $FFFFF6 and is not touched.
        a.label("soft_clr")
        a.abs_w(0x1439, RATIO)       # move.b ratio, d2
        a.w(0x0802)
        a.w(0x0007)                  # btst #7, d2
        a.bcc(0x66, "soft_cookie")
        a.w(0x7400)                  # moveq #0, d2
        a.label("soft_cookie")
        a.abs_w(0x1C39, RECHECK)     # move.b recheck, d6
        a.w(0x0C06)
        a.w(0x005A)
        a.bcc(0x67, "soft_wipe")
        a.w(0x7C00)                  # moveq #0, d6
        a.label("soft_wipe")
        a.w(0x7000)                  # moveq #0, d0
        a.w(0x41F9)
        a.l(0xFF0000)                # lea FF0000, a0
        a.w(0x323C)
        a.w(0x007B)                  # 124 longs, through FF01EF
        a.label("soft_lo")
        a.w(0x20C0)                  # move.l d0, (a0)+
        a.dbf(1, "soft_lo")
        a.w(0x4218)                  # clr.b (a0)+  FF01F0
        a.w(0x4218)                  # clr.b (a0)+  FF01F1
        a.w(0x10C2)                  # move.b d2, (a0)+
        a.w(0x4218)                  # clr.b (a0)+  ENTRY
        a.w(0x720A)                  # moveq #10, d1
        a.label("soft_mid")
        a.w(0x4218)
        a.dbf(1, "soft_mid")         # FF01F4..FF01FE
        a.w(0x4218)                  # clr.b (a0)+  FF01FF, a0 = FF0200
        a.w(0x323C)
        a.w(0x3F7C)                  # longs through FFFFF3
        a.label("soft_hi")
        a.w(0x20C0)
        a.dbf(1, "soft_hi")
        a.w(0x4218)                  # FFFFF4
        a.w(0x4218)                  # FFFFF5
        a.w(0x13C6)
        a.l(RECHECK)                 # move.b d6, recheck
        a.abs_w(0x4239, 0xFF0204)    # the instruction this hook replaced
        a.w(0x4EF9)
        a.l(0x185BD4)

        # Drawn height follows the road, then a crest or the end of a jump
        # reverses it in one frame. That reads as the rider bobbing. On a
        # reversal, take half of the step; a climb that keeps its direction
        # is stored whole. 125(a1) remembers the direction. Stock applies
        # the whole step. d0 is the target height.
        a.label("damp")
        a.w(0x3229)
        a.w(0x0002)                  # move.w 2(a1), d1
        a.w(0x9041)                  # sub.w d1, d0   delta
        a.w(0x3F00)                  # move.w d0, -(sp)
        a.w(0x7000)
        a.abs_w(0x1039, RATIO)
        a.w(0x0800)
        a.w(0x0007)
        a.bcc(0x67, "damp_full")
        a.w(0x0240)
        a.w(0x007F)
        a.w(0x0C40)
        a.w(16)
        a.bcc(0x67, "damp_full")
        a.w(0x1029)
        a.w(0x007D)                  # move.b 125(a1), d0
        a.w(0x0C00)
        a.w(1)
        a.bcc(0x67, "damp_opp")
        a.w(0x0C00)
        a.w(0x00FF)
        a.bcc(0x66, "damp_full")
        a.label("damp_opp")
        a.w(0x4880)                  # ext.w d0
        a.w(0x3217)                  # move.w (sp), d1
        a.w(0xB141)                  # eor.w d1, d0
        a.bcc(0x6A, "damp_full")     # bpl: same direction
        a.w(0x3217)
        a.w(0xE241)                  # asr.w #1, d1
        a.w(0x3E81)                  # move.w d1, (sp)
        a.label("damp_full")
        a.w(0x301F)                  # move.w (sp)+, d0
        a.w(0x4A40)                  # tst.w d0
        a.bcc(0x67, "damp_add")
        a.bcc(0x6B, "damp_neg")
        a.w(0x137C)
        a.w(0x0001)
        a.w(0x007D)                  # move.b #1, 125(a1)
        a.bcc(0x60, "damp_add")
        a.label("damp_neg")
        a.w(0x137C)
        a.w(0x00FF)
        a.w(0x007D)
        a.label("damp_add")
        a.w(0xD069)
        a.w(0x0002)                  # add.w 2(a1), d0
        a.w(0x3340)
        a.w(0x0002)                  # move.w d0, 2(a1)
        a.w(0x4E75)

        a.label("air_vis")
        a.w(0x2F00)                  # move.l d0, -(sp)
        a.w(0x2F01)                  # move.l d1, -(sp)
        a.w(0x3002)                  # move.w d2, d0
        a.bcc(0x61, "damp")
        a.w(0x221F)                  # move.l (sp)+, d1
        a.w(0x201F)                  # move.l (sp)+, d0
        a.w(0x08E9)
        a.w(0x0000)
        a.w(0x0068)                  # bset #0, 104(a1)
        a.w(0x4E75)

        a.label("ground_vis")
        a.w(0x2F00)
        a.w(0x2F01)
        a.w(0x2F02)                  # move.l d2, -(sp)
        a.w(0x3001)                  # move.w d1, d0
        a.bcc(0x61, "damp")
        a.w(0x241F)                  # move.l (sp)+, d2
        a.w(0x221F)
        a.w(0x201F)
        a.w(0x337C)
        a.w(0x0000)
        a.w(0x0032)                  # move.w #0, 50(a1)
        a.w(0x4E75)

    a.buf[calib_at:calib_at + 4] = a.labels["calib"].to_bytes(4, "big")
    a.buf[pause_jsr:pause_jsr + 4] = a.labels["calib"].to_bytes(4, "big")
    a.buf[race_jsr:race_jsr + 4] = a.labels["calib"].to_bytes(4, "big")
    a.buf[snap_at:snap_at + 4] = a.labels["snap"].to_bytes(4, "big")
    a.buf[est_at:est_at + 4] = a.labels["est_update"].to_bytes(4, "big")
    a.buf[setup_at:setup_at + 4] = a.labels["setup_done"].to_bytes(4, "big")
    if STOCK_LINES:
        a.buf[scale_at:scale_at + 4] = a.labels["scale_now"].to_bytes(4, "big")
        a.buf[charge_at:charge_at + 4] = a.labels["charge"].to_bytes(4, "big")
        a.buf[wall_abs:wall_abs + 4] = a.labels["wall"].to_bytes(4, "big")
        a.buf[wall_chg:wall_chg + 4] = a.labels["wall"].to_bytes(4, "big")

    code = a.finish()
    return code, a.labels


def assemble(kind):
    # ti: his content, our physics, our split-screen queue.
    # ti-split: the same, and his split-screen uploader stays in place.
    ti = kind != "base"
    his_split = kind == "ti-split"
    if kind == "base":
        org, ips_path, out_rom, out_ips = ORG, None, OUT_ROM, OUT_IPS
    elif kind in ("ti", "ti-split"):
        org = 0x1FCB44
        ips_path = TI_DIR / "ROAD_RASH_3_Improvement_v1.04.ips"
        if his_split:
            out_rom = OUT_DIR / "Road Rash 3 (physics+ti-split).gen"
            out_ips = OUT_DIR / "road-rash-3-physics-ti-split.ips"
        else:
            out_rom = OUT_DIR / "Road Rash 3 (physics+ti).gen"
            out_ips = OUT_DIR / "road-rash-3-physics-ti.ips"
    else:
        org, ips_path, out_rom, out_ips = VARIANTS[kind]
    vanilla = SRC.read_bytes()
    # physics+ti puts his frame back to stock, so the vblank has to count
    # the FPS period. The other two keep his frame, which counts it already.
    code, labels = build_base(org, cd=ti, fps_tick=(kind == "ti"))
    rom = bytearray(vanilla)
    if sega_checksum(rom) != 0x388AA63F:
        raise SystemExit("source ROM checksum does not match the game")
    if ti:
        apply_ips(rom, ips_path)
        # His turn step is 6/frame-length. Ours scales that same product
        # by the calibrated clock.
        rom[0x1E1E4:0x1E1F4] = vanilla[0x1E1E4:0x1E1F4]
        rom[0x1E514:0x1E518] = vanilla[0x1E514:0x1E518]
        if not his_split:
            rom[0x12D7C:0x12E98] = vanilla[0x12D7C:0x12E98]
    if kind == "base":
        ti_rom = bytearray(vanilla)
        apply_ips(ti_rom, TI_DIR / "ROAD_RASH_3_Improvement_v1.04.ips")
        # His one-vblank upload. The picture is sent with the display off,
        # so the sliced queue and its top-of-screen flicker are not used.
        rom[0x12D7C:0x12E98] = ti_rom[0x12D7C:0x12E98]
        rom[0x1FC76A:0x1FCAC6] = ti_rom[0x1FC76A:0x1FCAC6]

    def expect(addr, data):
        got = bytes(rom[addr:addr + len(data)])
        if got != data:
            raise SystemExit(f"{addr:#x}: {got.hex()} != {data.hex()}")

    def patch(addr, data):
        rom[addr:addr + len(data)] = data

    if ti:
        expect(0x21B6, bytes.fromhex("48E7FFFE08F9000800A12000"))
        patch(0x21B6, bytes.fromhex("4EF9") + labels["vbl_entry"].to_bytes(4, "big") + b"\x4E\x71" * 3)
        if any(b != 0xFF for b in rom[org:org + len(code)]):
            raise SystemExit("Ti left no free space at the code origin")
    else:
        expect(0x21B6, bytes.fromhex("48E7FFFE207900FF0226"))
        patch(0x21B6, bytes.fromhex("4EF9") + labels["vbl_entry"].to_bytes(4, "big") + b"\x4E\x71\x4E\x71")
    expect(0x221A, bytes.fromhex("4CDF7FFF4E73"))
    patch(0x221A, bytes.fromhex("4EF9") + labels["vbl_leave"].to_bytes(4, "big"))
    # base and ti-split use his frame, so the stock dispatcher is gone.
    if kind == "ti":
        expect(0x12D7C, bytes.fromhex("33FC813400C00004"))
        patch(0x12D7C, bytes.fromhex("4EB9") + labels["pause_cal"].to_bytes(4, "big") + b"\x4E\x71")
        # 18 bytes. One more nop eats the first bra of the jump table.
        expect(0x12E4C, bytes.fromhex("303900FF3236E54041F900012E5E4EF00000"))
        patch(0x12E4C, bytes.fromhex("4EF9") + labels["two_dma"].to_bytes(4, "big") + b"\x4E\x71" * 6)
    else:
        expect(0x1FCA0A, bytes.fromhex("4EF900012FA4"))
        patch(0x1FCA0A, bytes.fromhex("4EF9") + labels["fps_full"].to_bytes(4, "big"))
        expect(0x1FCAC0, bytes.fromhex("4EF900012FD2"))
        patch(0x1FCAC0, bytes.fromhex("4EF9") + labels["fps_part"].to_bytes(4, "big"))
    expect(0x18296, bytes.fromhex("4EB9000182BC"))
    patch(0x18296, bytes.fromhex("4EB9") + labels["cam_p2"].to_bytes(4, "big"))
    expect(0x182E2, bytes.fromhex("91B900FF36E0"))
    patch(0x182E2, bytes.fromhex("4EB9") + labels["scale_sub"].to_bytes(4, "big"))
    expect(0x182FE, bytes.fromhex("D1B900FF36E0"))
    patch(0x182FE, bytes.fromhex("4EB9") + labels["scale_add"].to_bytes(4, "big"))
    expect(0x1E47A, bytes.fromhex("20290026E088"))
    patch(0x1E47A, bytes.fromhex("4EB9") + labels["scale_hdg"].to_bytes(4, "big"))
    expect(0x1E418, bytes.fromhex("20290026E088"))
    patch(0x1E418, bytes.fromhex("4EB9") + labels["scale_hdg"].to_bytes(4, "big"))
    for addr in (0x1E472, 0x1E540):
        expect(addr, bytes.fromhex("33400022D1514E75"))
        patch(addr, bytes.fromhex("4EB9") + labels["lat_step"].to_bytes(4, "big"))
    for i, addr in enumerate((0x18836, 0x19262)):
        expect(addr, bytes.fromhex("323900FF09D4"))
        patch(addr, bytes.fromhex("4EB9") + labels[f"hill_lat{i}"].to_bytes(4, "big"))
    expect(0x1E764, bytes.fromhex("22290026E089C3C6E081C3FC0300E081"))
    patch(0x1E764, bytes.fromhex("4EB9") + labels["scale_lane"].to_bytes(4, "big") + b"\x4E\x71" * 5)
    expect(0x1E712, bytes.fromhex("DF69003A30290006"))
    patch(0x1E712, bytes.fromhex("4EB9") + labels["height_add"].to_bytes(4, "big") + b"\x4E\x71")
    expect(0x1E878, bytes.fromhex("3342000208E900000068"))
    patch(0x1E878, bytes.fromhex("4EB9") + labels["air_vis"].to_bytes(4, "big") + b"\x4E\x71\x4E\x71")
    expect(0x1E894, bytes.fromhex("33410002337C00000032"))
    patch(0x1E894, bytes.fromhex("4EB9") + labels["ground_vis"].to_bytes(4, "big") + b"\x4E\x71\x4E\x71")
    expect(0x1E798, bytes.fromhex("3341003CD269003A"))
    patch(0x1E798, bytes.fromhex("4EB9") + labels["height_join"].to_bytes(4, "big") + b"\x4E\x71")
    expect(0x1E7F8, bytes.fromhex("303900FF098A"))
    patch(0x1E7F8, bytes.fromhex("4EB9") + labels["curve_base"].to_bytes(4, "big"))
    expect(0x1E82C, bytes.fromhex("3340002C88C3"))
    patch(0x1E82C, bytes.fromhex("4EB9") + labels["kick_keep"].to_bytes(4, "big"))
    expect(0x1E872, bytes.fromhex("B2426A00001E"))
    patch(0x1E872, bytes.fromhex("4EF9") + labels["stick"].to_bytes(4, "big"))
    expect(0x1A3C0, bytes.fromhex("488367000016"))
    patch(0x1A3C0, bytes.fromhex("4EB9") + labels["road_slope"].to_bytes(4, "big"))
    expect(0x1A5C4, bytes.fromhex("488367000016"))
    patch(0x1A5C4, bytes.fromhex("4EB9") + labels["road_slope"].to_bytes(4, "big"))
    expect(0x1A50A, bytes.fromhex("4882D64251C8FEF8"))
    patch(0x1A50A, bytes.fromhex("4EB9") + labels["slope_step_a"].to_bytes(4, "big") + b"\x4E\x71")
    expect(0x1A69C, bytes.fromhex("4882D64251C8FF56"))
    patch(0x1A69C, bytes.fromhex("4EB9") + labels["slope_step_b"].to_bytes(4, "big") + b"\x4E\x71")
    expect(0x1DD3A, bytes.fromhex("343900FF1BDCC0C223400026"))
    patch(0x1DD3A, bytes.fromhex("4EB9") + labels["coast_stop"].to_bytes(4, "big") + b"\x4E\x71" * 3)
    expect(0x185EE8, bytes.fromhex("4EB900024BDE"))
    patch(0x185EE8, bytes.fromhex("4EB9") + labels["pause_race"].to_bytes(4, "big"))
    expect(0x185BCE, bytes.fromhex("423900FF0204"))
    patch(0x185BCE, bytes.fromhex("4EF9") + labels["soft_clr"].to_bytes(4, "big"))

    expect(0x13D76, bytes.fromhex("207900FF278A"))
    patch(0x13D76, bytes.fromhex("4EB9") + labels["phys_gate"].to_bytes(4, "big"))
    expect(0x13088, bytes.fromhex("247900FF278E"))
    patch(0x13088, bytes.fromhex("4EB9") + labels["cam_release"].to_bytes(4, "big"))
    expect(0x131D6, bytes.fromhex("247900FF278A34BC0001"))
    patch(0x131D6, bytes.fromhex("4EB9") + labels["arm_pic"].to_bytes(4, "big") + b"\x4E\x71\x4E\x71")
    expect(0x13D7E, bytes.fromhex("6600012C"))
    patch(0x13D7E, b"\x4E\x71\x4E\x71")
    patch(org, code)

    if ti:
        # Ti replaced the checksum routine. The long at 0x1FA080 is his code.
        total = None
    else:
        total = sega_checksum(rom)
        patch(0x1FA080, total.to_bytes(4, "big"))
        if sega_checksum(rom) != total:
            raise SystemExit("checksum constant is inside the summed region")

    rom[0x18E:0x190] = header_checksum(rom).to_bytes(2, "big")
    if header_checksum(rom) != int.from_bytes(rom[0x18E:0x190], "big"):
        raise SystemExit("header checksum does not stick")
    if not ti and sega_checksum(rom) != total:
        raise SystemExit("header checksum landed inside the internal sum")

    out_rom.write_bytes(rom)
    nranges, nips = write_ips(SRC.read_bytes(), rom, out_ips)
    print(f"{kind} code {org:#x}..{org + len(code):#x} ({len(code)} bytes)")
    print(f"  rom {out_rom.name}  ips {out_ips.name} ({nips} bytes, {nranges} ranges)")
    if total is not None:
        print(f"  internal checksum {total:#010x}")


def main():
    p = argparse.ArgumentParser(description="Build Road Rash 3 physics ROM and IPS")
    p.add_argument("--rom", type=Path, help="vanilla ROM (.gen)")
    p.add_argument("--out", type=Path, help="output directory (default: script directory)")
    p.add_argument("--ti-dir", type=Path,
                   help="directory with ROAD_RASH_3_Improvement_v1.04.ips")
    args = p.parse_args()
    configure(args.rom, args.out, args.ti_dir)
    if not SRC.is_file():
        raise SystemExit(f"vanilla ROM not found: {SRC}")
    assemble("base")
    assemble("ti")
    assemble("ti-split")


if __name__ == "__main__":
    main()
