"""
tm_enigma.py

Turing Machine simulator implemented as an Enigma-like 3-rotor cipher UI using curses (NO tkinter).

What it does:
- Implements a basic TM (tape + head + state + delta)
- Uses an Enigma-like 3-rotor cipher as the "computation" the TM performs on the tape
- Visualizes:
  - rotor windows + wheels
  - plugboard pairs
  - keyboard input highlight + lampboard output highlight
  - TM tape window + head
  - per-step signal path log
  - wiring tables panel

Controls:
  L / X  Load/Reset TM (message + current settings + start positions)
  S      Step
  R      Run/Stop (auto-run)
  +      Faster steps
  -      Slower steps
  E      Edit message
  P      Edit start positions (Grundstellung) e.g. "A A A"
  G      Edit ring settings (Ringstellung) e.g. "A A A"
  O      Edit rotor order e.g. "I II III"
  B      Edit plugboard pairs e.g. "AQ EP MT"
  C      Copy output into message (does not auto-reload)
  D      Decrypt (message := output, then reload with same settings)
  [ / ]  Scroll wiring panel
  Up/Down Scroll signal log
  Q      Quit

Notes:
- Alphabet A–Z plus space.
- Space is pass-through, but rotors still step (keeps stepping behavior consistent).
- Enigma is symmetric: decrypt == encrypt again with identical settings + start positions.
- TM stores rotor positions inside the state string: q_scan|LL|MM|RR

Disclaimer: AI was used to help with the visuals to improve user experience.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Iterable, List
import curses
import time
import re

# ============================================================
# 1) TURING MACHINE CORE
# ============================================================

Move = str
Transition = Tuple[str, str, Move]


@dataclass
class TMConfig:
    blank: str = "_"
    max_steps: int = 500000
    window: int = 22  # tape window radius


class Tape:
    """Sparse tape: store only non-blank cells in a dict."""
    def __init__(self, initial: str, blank: str = "_"):
        self.blank = blank
        self.cells: Dict[int, str] = {}
        for i, ch in enumerate(initial):
            if ch != blank:
                self.cells[i] = ch

    def read(self, i: int) -> str:
        return self.cells.get(i, self.blank)

    def write(self, i: int, ch: str):
        if ch == self.blank:
            self.cells.pop(i, None)
        else:
            self.cells[i] = ch

    def content(self, start: int = 0) -> str:
        """Return content from start to last non-blank cell."""
        if not self.cells:
            return ""
        last = max(self.cells.keys())
        return "".join(self.read(i) for i in range(start, last + 1))

    def snapshot(self, head: int, window: int) -> Tuple[str, str, int]:
        """Return a printable window around the head plus caret line."""
        left = head - window
        right = head + window
        chars = [self.read(i) for i in range(left, right + 1)]
        tape_line = "".join(chars)
        caret = [" "] * len(chars)
        caret[head - left] = "^"
        return tape_line, "".join(caret), left


class TuringMachine:
    """
    A minimal TM:
    - tape (infinite, sparse)
    - head position
    - current state
    - transition function delta(state, symbol) -> (new_state, write, move)
    """
    def __init__(
        self,
        tape: Tape,
        start_state: str,
        accept_states: Iterable[str],
        reject_states: Iterable[str],
        delta: Callable[[str, str], Optional[Transition]],
        head: int = 0,
        config: TMConfig = TMConfig(),
        hook: Optional[Callable[[dict], None]] = None,
    ):
        self.tape = tape
        self.state = start_state
        self.accept = set(accept_states)
        self.reject = set(reject_states)
        self.delta = delta
        self.head = head
        self.steps = 0
        self.config = config
        self.hook = hook

    def halted(self) -> bool:
        return self.state in self.accept or self.state in self.reject or self.steps >= self.config.max_steps

    def step(self) -> bool:
        """Run exactly one TM transition."""
        if self.halted():
            return False

        sym = self.tape.read(self.head)
        trans = self.delta(self.state, sym)

        # Undefined transition -> reject
        if trans is None:
            self.state = next(iter(self.reject), "q_reject")
            return False

        new_state, write_sym, move = trans
        self.tape.write(self.head, write_sym)

        if move == "L":
            self.head -= 1
        elif move == "R":
            self.head += 1
        elif move == "S":
            pass
        else:
            raise ValueError("Invalid move")

        self.state = new_state
        self.steps += 1
        return True


# ============================================================
# 2) ENIGMA MODEL (3 rotors + rings + plugboard)
# ============================================================

ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IDX = {ch: i for i, ch in enumerate(ALPH)}


def normalize(s: str) -> str:
    return s.upper()


def parse_triplet_letters(s: str) -> Tuple[int, int, int]:
    """Parse 'A A A' -> (0,0,0)."""
    parts = normalize(s).replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Need 3 letters like: A A A")
    vals = []
    for p in parts:
        if p not in ALPH:
            raise ValueError("Letters must be A–Z")
        vals.append(IDX[p])
    return (vals[0], vals[1], vals[2])


def parse_rotor_order(s: str) -> Tuple[str, str, str]:
    """Parse 'I II III' etc."""
    parts = normalize(s).replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Need 3 rotor names like: I II III")
    allowed = {"I", "II", "III", "IV", "V"}
    for p in parts:
        if p not in allowed:
            raise ValueError("Rotor names must be I, II, III, IV, or V")
    return (parts[0], parts[1], parts[2])  # left, mid, right


def parse_plugboard(s: str) -> Tuple[Tuple[str, str], ...]:
    """
    Accepts formats like:
      "AQ EP MT"
      "A-Q E-P M-T"
      "AQ,EP,MT"
    Each letter can appear at most once.
    """
    s = normalize(s).strip()
    if not s:
        return ()
    tokens = re.split(r"[\s,]+", s)
    pairs = []
    used = set()

    for t in tokens:
        t = t.replace("-", "")
        if not t:
            continue
        if len(t) != 2 or t[0] not in ALPH or t[1] not in ALPH or t[0] == t[1]:
            raise ValueError("Plugboard pairs must be two different letters like AQ EP MT")
        a, b = t[0], t[1]
        if a in used or b in used:
            raise ValueError("A letter can appear in at most one plugboard pair")
        used.add(a)
        used.add(b)
        pairs.append((a, b))

    return tuple(pairs)


@dataclass
class EnigmaSettings:
    rotor_order: Tuple[str, str, str] = ("I", "II", "III")  # left, mid, right
    start_positions: Tuple[int, int, int] = (0, 0, 0)       # Grundstellung
    ring_settings: Tuple[int, int, int] = (0, 0, 0)         # Ringstellung
    plugboard_pairs: Tuple[Tuple[str, str], ...] = ()


class Enigma:
    """
    Enigma-like machine:
    - 3 rotors
    - reflector B
    - ring settings (Ringstellung)
    - plugboard pairs (Steckerbrett)
    - stepping with a common didactic double-step rule
    """

    ROTORS = {
        "I":   ("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q"),
        "II":  ("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E"),
        "III": ("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V"),
        "IV":  ("ESOVPZJAYQUIRHXLNFTGKDCMWB", "J"),
        "V":   ("VZBRGITYUPSDNHLXAWMJQOFECK", "Z"),
    }
    REFLECTOR_B = "YRUHQSLDPXNGOKMIEBFZCWVJAT"

    def __init__(
        self,
        rotor_order: Tuple[str, str, str],
        positions: Tuple[int, int, int],
        rings: Tuple[int, int, int],
        plug_pairs: Tuple[Tuple[str, str], ...]
    ):
        self.left_name, self.mid_name, self.right_name = rotor_order
        self.left_pos, self.mid_pos, self.right_pos = positions
        self.left_ring, self.mid_ring, self.right_ring = rings

        self.left_w, self.left_notch = self.ROTORS[self.left_name]
        self.mid_w, self.mid_notch = self.ROTORS[self.mid_name]
        self.right_w, self.right_notch = self.ROTORS[self.right_name]

        # Inverse wirings for reverse pass
        self.left_inv = self._invert(self.left_w)
        self.mid_inv = self._invert(self.mid_w)
        self.right_inv = self._invert(self.right_w)

        # Plugboard mapping
        self.plug = self._build_plugboard(plug_pairs)

    def _build_plugboard(self, pairs: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
        m = {ch: ch for ch in ALPH}
        for a, b in pairs:
            m[a], m[b] = b, a
        return m

    def _invert(self, wiring: str) -> str:
        inv = ["?"] * 26
        for i, ch in enumerate(wiring):
            inv[IDX[ch]] = ALPH[i]
        return "".join(inv)

    def get_positions(self) -> Tuple[int, int, int]:
        return (self.left_pos, self.mid_pos, self.right_pos)

    def _step_rotors(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], str]:
        """
        Stepping rule:
          - If middle rotor at notch: left + middle step (double-step effect)
          - Else if right rotor at notch: middle steps
          - Right always steps
        """
        before = self.get_positions()
        mid_at_notch = (ALPH[self.mid_pos] == self.mid_notch)
        right_at_notch = (ALPH[self.right_pos] == self.right_notch)

        note = []
        if mid_at_notch:
            self.left_pos = (self.left_pos + 1) % 26
            self.mid_pos = (self.mid_pos + 1) % 26
            note.append("double-step (mid notch): L+M step")
        elif right_at_notch:
            self.mid_pos = (self.mid_pos + 1) % 26
            note.append("right notch: M steps")

        self.right_pos = (self.right_pos + 1) % 26
        note.append("R steps")

        after = self.get_positions()
        return before, after, "; ".join(note)

    def _rotor_fwd(self, ch: str, wiring: str, pos: int, ring: int) -> str:
        # index = (c + pos - ring) mod 26 ; out = (wiring[index] - pos + ring) mod 26
        i = (IDX[ch] + pos - ring) % 26
        wired = wiring[i]
        o = (IDX[wired] - pos + ring) % 26
        return ALPH[o]

    def _rotor_rev(self, ch: str, inv_wiring: str, pos: int, ring: int) -> str:
        i = (IDX[ch] + pos - ring) % 26
        wired = inv_wiring[i]
        o = (IDX[wired] - pos + ring) % 26
        return ALPH[o]

    def encipher_char(self, ch: str) -> Tuple[str, dict]:
        """
        Encrypt one character + return a trace dictionary for visualization/logging.
        """
        trace = {
            "input": ch,
            "pos_before": None,
            "pos_after": None,
            "step_note": "",
            "path": [],
            "lit": None,  # output lamp letter (A–Z) or None
        }

        pos_before, pos_after, note = self._step_rotors()
        trace["pos_before"] = pos_before
        trace["pos_after"] = pos_after
        trace["step_note"] = note

        # Space passes through (still steps rotors)
        if ch == " ":
            trace["path"] = ["SPACE (pass-through; rotors still stepped)"]
            trace["lit"] = None
            return " ", trace

        # Non A–Z passes through (still steps rotors)
        if ch not in ALPH:
            trace["path"] = ["NON-AZ (pass-through; rotors still stepped)"]
            trace["lit"] = None
            return ch, trace

        # Plugboard in
        p0 = self.plug.get(ch, ch)
        trace["path"].append(f"Plugboard IN: {ch} -> {p0}")

        # Forward: Right -> Mid -> Left
        r1 = self._rotor_fwd(p0, self.right_w, self.right_pos, self.right_ring)
        trace["path"].append(f"Rotor {self.right_name} FWD (pos {ALPH[self.right_pos]} ring {ALPH[self.right_ring]}): {p0} -> {r1}")

        r2 = self._rotor_fwd(r1, self.mid_w, self.mid_pos, self.mid_ring)
        trace["path"].append(f"Rotor {self.mid_name} FWD (pos {ALPH[self.mid_pos]} ring {ALPH[self.mid_ring]}): {r1} -> {r2}")

        r3 = self._rotor_fwd(r2, self.left_w, self.left_pos, self.left_ring)
        trace["path"].append(f"Rotor {self.left_name} FWD (pos {ALPH[self.left_pos]} ring {ALPH[self.left_ring]}): {r2} -> {r3}")

        # Reflector
        ref = self.REFLECTOR_B[IDX[r3]]
        trace["path"].append(f"Reflector B: {r3} -> {ref}")

        # Reverse: Left -> Mid -> Right
        b3 = self._rotor_rev(ref, self.left_inv, self.left_pos, self.left_ring)
        trace["path"].append(f"Rotor {self.left_name} REV (pos {ALPH[self.left_pos]} ring {ALPH[self.left_ring]}): {ref} -> {b3}")

        b2 = self._rotor_rev(b3, self.mid_inv, self.mid_pos, self.mid_ring)
        trace["path"].append(f"Rotor {self.mid_name} REV (pos {ALPH[self.mid_pos]} ring {ALPH[self.mid_ring]}): {b3} -> {b2}")

        b1 = self._rotor_rev(b2, self.right_inv, self.right_pos, self.right_ring)
        trace["path"].append(f"Rotor {self.right_name} REV (pos {ALPH[self.right_pos]} ring {ALPH[self.right_ring]}): {b2} -> {b1}")

        # Plugboard out
        out = self.plug.get(b1, b1)
        trace["path"].append(f"Plugboard OUT: {b1} -> {out}")
        trace["lit"] = out  # always A–Z here
        return out, trace


# ============================================================
# 3) ENIGMA-AS-TM (rotor positions stored in TM state string)
# ============================================================

def encode_state(l: int, m: int, r: int) -> str:
    return f"q_scan|{l:02d}|{m:02d}|{r:02d}"


def decode_state(state: str) -> Tuple[str, int, int, int]:
    parts = state.split("|")
    if len(parts) != 4:
        return state, 0, 0, 0
    return parts[0], int(parts[1]), int(parts[2]), int(parts[3])


def tape_result(tape: Tape, blank: str) -> str:
    out = tape.content(0)
    if blank in out:
        out = out.split(blank, 1)[0]
    return out


def build_enigma_tm(
    message: str,
    settings: EnigmaSettings,
    config: TMConfig,
    hook: Optional[Callable[[dict], None]] = None,
) -> TuringMachine:
    """
    TM behavior:
      - Scan left-to-right on tape
      - Replace each symbol with its Enigma output
      - Move right
      - Halt accept when blank is reached
    """
    msg = normalize(message)
    tape = Tape(initial=msg + config.blank, blank=config.blank)

    tm_ref = {"tm": None}

    def delta(state: str, sym: str) -> Optional[Transition]:
        q, lpos, mpos, rpos = decode_state(state)
        if q != "q_scan":
            return None

        if sym == config.blank:
            if hook:
                hook({"event": "halt", "halt_state": "q_accept"})
            return ("q_accept", sym, "S")

        # Reconstruct Enigma using rotor positions stored in the TM state
        e = Enigma(
            rotor_order=settings.rotor_order,
            positions=(lpos, mpos, rpos),
            rings=settings.ring_settings,
            plug_pairs=settings.plugboard_pairs,
        )

        out_ch, trace = e.encipher_char(sym)
        nl, nm, nr = e.get_positions()
        new_state = encode_state(nl, nm, nr)

        # Let UI observe the step
        if hook and tm_ref["tm"] is not None:
            tm = tm_ref["tm"]
            tape_line, head_line, left_idx = tape.snapshot(tm.head, tm.config.window)
            hook({
                "event": "step",
                "read": sym,
                "write": out_ch,
                "lit": trace.get("lit"),
                "state": state,
                "new_state": new_state,
                "head": tm.head,
                "steps": tm.steps,
                "pos_before": trace["pos_before"],
                "pos_after": trace["pos_after"],
                "step_note": trace.get("step_note", ""),
                "path": trace["path"],
                "tape_line": tape_line,
                "head_line": head_line,
                "tape_left": left_idx,
            })

        return (new_state, out_ch, "R")

    start_state = encode_state(*settings.start_positions)
    tm = TuringMachine(
        tape=tape,
        start_state=start_state,
        accept_states={"q_accept"},
        reject_states={"q_reject"},
        delta=delta,
        head=0,
        config=config,
        hook=hook,
    )
    tm_ref["tm"] = tm
    return tm


# ============================================================
# 4) CURSES UI
# ============================================================

# Staggered QWERTY-style layout: (row_letters, left_offset_spaces)
KB_LAYOUT = [
    ("QWERTYUIOP", 0),
    ("ASDFGHJKL",  2),
    ("ZXCVBNM",    4),
]


def safe_add(stdscr, y, x, s, attr=0):
    """Write safely within screen bounds (not box-aware)."""
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h:
        return
    if x < 0:
        s = s[-x:]
        x = 0
    if x >= w:
        return
    s = s[: max(0, w - x)]
    if s:
        stdscr.addstr(y, x, s, attr)


def draw_box(stdscr, y, x, h, w, title="", attr=0):
    """Draw a simple ASCII box."""
    if h < 2 or w < 2:
        return
    safe_add(stdscr, y, x, "+" + "-"*(w-2) + "+", attr)
    for i in range(1, h-1):
        safe_add(stdscr, y+i, x, "|", attr)
        safe_add(stdscr, y+i, x+w-1, "|", attr)
    safe_add(stdscr, y+h-1, x, "+" + "-"*(w-2) + "+", attr)
    if title:
        t = f" {title} "
        if len(t) < w-2:
            safe_add(stdscr, y, x+2, t, attr)


def prompt_line(stdscr, prompt: str, default: str, attr: int):
    """Bottom-line prompt for editing."""
    curses.echo()
    stdscr.nodelay(False)
    h, w = stdscr.getmaxyx()
    y = h - 2
    safe_add(stdscr, y, 0, " " * (w-1), attr)
    msg = f"{prompt} (default: {default}) > "
    safe_add(stdscr, y, 0, msg, attr)
    stdscr.refresh()
    s = stdscr.getstr(y, min(w-2, len(msg))).decode("utf-8", errors="ignore")
    curses.noecho()
    stdscr.nodelay(True)
    return s.strip() if s.strip() else default


def rotor_wheel_strip(pos: int) -> str:
    """Show letters around the rotor window letter (visual wheel)."""
    letters = [ALPH[(pos + d) % 26] for d in (-3, -2, -1, 0, 1, 2, 3)]
    return " ".join(letters)


def wiring_panel_lines(settings: EnigmaSettings) -> List[str]:
    """Right panel: show configuration + rotor/reflector wiring tables."""
    lines = []
    pairs = " ".join([a+b for (a, b) in settings.plugboard_pairs]) if settings.plugboard_pairs else "(none)"
    lines.append("CONFIG SUMMARY")
    lines.append(f"Rotor order (L M R): {settings.rotor_order[0]} {settings.rotor_order[1]} {settings.rotor_order[2]}")
    lines.append(f"Ring settings (L M R): {ALPH[settings.ring_settings[0]]} {ALPH[settings.ring_settings[1]]} {ALPH[settings.ring_settings[2]]}")
    lines.append(f"Plugboard pairs: {pairs}")
    lines.append("")
    lines.append("WIRING TABLES (A->?)")
    lines.append("Index:      " + " ".join(ALPH))
    for name in settings.rotor_order:
        wiring, notch = Enigma.ROTORS[name]
        lines.append(f"Rotor {name} (notch {notch}): " + " ".join(wiring))
    lines.append("Reflector B: " + " ".join(Enigma.REFLECTOR_B))
    lines.append("")
    lines.append("Note: mapping uses offsets from BOTH rotor window position and ring setting.")
    return lines


def _render_key_row(row: str, offset_spaces: int, highlight: Optional[str]) -> str:
    """Build one keyboard row string with staggered offset."""
    keys = []
    for ch in row:
        if highlight == ch:
            keys.append(f"[{ch}]")
        else:
            keys.append(f" {ch} ")
    return (" " * offset_spaces) + " ".join(keys)


def draw_keyboard_and_lamps_in_box(
    stdscr,
    box_y: int,
    box_x: int,
    box_h: int,
    box_w: int,
    start_y: int,
    lit_in: Optional[str],
    lit_out: Optional[str],
    A_NORM,
    A_KEY,
    A_LAMP
):
    """
    Draw keyboard + lampboard inside a box, never touching borders.
    Also clips to the available inner width and inner height.
    """
    inner_x = box_x + 2
    inner_w = max(1, box_w - 4)

    # Inner vertical boundaries
    top_limit = box_y + 1
    bot_limit = box_y + box_h - 2  # last drawable row inside box

    y = start_y
    if y < top_limit:
        y = top_limit

    def add_line(ypos: int, text: str, attr: int):
        if ypos < top_limit or ypos > bot_limit:
            return
        safe_add(stdscr, ypos, inner_x, text[:inner_w], attr)

    add_line(y, "KEYBOARD", A_NORM | curses.A_BOLD)
    y += 1

    # Keyboard rows
    for row, off in KB_LAYOUT:
        line = _render_key_row(row, off, lit_in)
        # Center horizontally within inner width
        x = inner_x + max(0, (inner_w - len(line)) // 2)
        if y >= top_limit and y <= bot_limit:
            safe_add(stdscr, y, x, line[:inner_w], A_KEY)
        y += 1

    y += 1
    add_line(y, "LAMPBOARD", A_NORM | curses.A_BOLD)
    y += 1

    # Lamp rows
    for row, off in KB_LAYOUT:
        line = _render_key_row(row, off, lit_out)
        x = inner_x + max(0, (inner_w - len(line)) // 2)
        if y >= top_limit and y <= bot_limit:
            safe_add(stdscr, y, x, line[:inner_w], A_LAMP)
        y += 1


class UIState:
    def __init__(self):
        self.settings = EnigmaSettings()
        self.message = "HELLO WORLD"
        self.tm: Optional[TuringMachine] = None

        self.running = False
        self.speed_sps = 25

        self.path_log: List[str] = []
        self.scroll_log = 0
        self.scroll_wiring = 0

        self.last_in: Optional[str] = None
        self.last_out: Optional[str] = None
        self.lamp_flash_until = 0.0

        self.status = "Loaded. S step, R run. D decrypt. O rotors, G rings, B plugboard."


def ui_loop(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)

    curses.start_color()
    curses.use_default_colors()

    # Dark-mode friendly colors
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # title
    curses.init_pair(2, curses.COLOR_WHITE, -1)    # normal
    curses.init_pair(3, curses.COLOR_YELLOW, -1)   # status
    curses.init_pair(4, curses.COLOR_GREEN, -1)    # lamp
    curses.init_pair(5, curses.COLOR_RED, -1)      # error
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # keyboard highlight

    A_TITLE = curses.color_pair(1) | curses.A_BOLD
    A_NORM  = curses.color_pair(2)
    A_STAT  = curses.color_pair(3)
    A_OK    = curses.color_pair(4) | curses.A_BOLD
    A_BAD   = curses.color_pair(5) | curses.A_BOLD
    A_KEY   = curses.color_pair(6) | curses.A_BOLD

    config = TMConfig(blank="_", max_steps=500000, window=22)
    ui = UIState()

    # -----------------------------
    # Hook: called every TM step to update UI trace state
    # FIXED: type-checks prevent None-in-string errors.
    # -----------------------------
    def hook(info: dict):
        if info.get("event") != "step":
            return

        rd = info.get("read")
        ui.last_in = rd if isinstance(rd, str) and rd in ALPH else None

        lit = info.get("lit")
        ui.last_out = lit if isinstance(lit, str) and lit in ALPH else None
        if ui.last_out:
            ui.lamp_flash_until = time.time() + 0.25

        before = info["pos_before"]
        after = info["pos_after"]
        note = info.get("step_note", "")

        header = (
            f"[{info['steps']:05d}] {info['read']!r}->{info['write']!r} "
            f"| {ALPH[before[0]]}{ALPH[before[1]]}{ALPH[before[2]]}"
            f" -> {ALPH[after[0]]}{ALPH[after[1]]}{ALPH[after[2]]}"
        )
        if note:
            header += f" | {note}"

        ui.path_log.append(header)
        for p in info["path"]:
            ui.path_log.append("    " + p)
        ui.path_log.append("")

    def load_tm(message: Optional[str] = None):
        """Reload the TM with current message + settings."""
        ui.running = False
        msg = normalize(message if message is not None else ui.message)
        ui.message = msg
        ui.tm = build_enigma_tm(ui.message, ui.settings, config, hook=hook)

        ui.path_log = []
        ui.scroll_log = 0
        ui.scroll_wiring = 0
        ui.last_in = None
        ui.last_out = None
        ui.lamp_flash_until = 0.0

        ui.status = "Loaded. S step / R run. D decrypt with same settings."

    def current_output() -> str:
        if not ui.tm:
            return ""
        return tape_result(ui.tm.tape, ui.tm.config.blank)

    load_tm()
    last_tick = time.time()

    while True:
        # Auto-run stepping
        if ui.running and ui.tm and not ui.tm.halted():
            now = time.time()
            interval = 1.0 / max(1, ui.speed_sps)
            if now - last_tick >= interval:
                ui.tm.step()
                last_tick = now
            else:
                time.sleep(0.001)
        else:
            time.sleep(0.01)

        # -----------------------------
        # DRAW
        # -----------------------------
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        safe_add(stdscr, 0, 2,
                 "ENIGMA (Terminal) — 3 Rotors + Rings + Plugboard — implemented as a Turing Machine"[:w-4],
                 A_TITLE)

        pairs_str = " ".join([a+b for (a, b) in ui.settings.plugboard_pairs]) if ui.settings.plugboard_pairs else "(none)"
        safe_add(stdscr, 1, 2, f"Rotors (L M R): {ui.settings.rotor_order[0]} {ui.settings.rotor_order[1]} {ui.settings.rotor_order[2]}"[:w-4], A_NORM)
        safe_add(stdscr, 2, 2, f"Start pos (Grundstellung): {ALPH[ui.settings.start_positions[0]]} {ALPH[ui.settings.start_positions[1]]} {ALPH[ui.settings.start_positions[2]]}"[:w-4], A_NORM)
        safe_add(stdscr, 3, 2, f"Ring (Ringstellung):       {ALPH[ui.settings.ring_settings[0]]} {ALPH[ui.settings.ring_settings[1]]} {ALPH[ui.settings.ring_settings[2]]}    Plugboard: {pairs_str}"[:w-4], A_NORM)
        if w >= 28:
            safe_add(stdscr, 1, w-24, f"Speed: {ui.speed_sps:3d} s/s", A_NORM)

        # Columns
        left_w = max(46, w // 3)
        mid_w  = max(36, w // 3)
        right_w = w - left_w - mid_w - 6
        if right_w < 30:
            right_w = 30
            mid_w = max(30, w - left_w - right_w - 6)

        top_y = 5

        # Make left panel tall enough, but never exceed the screen
        left_h = min(22, max(10, h - top_y - 3))

        draw_box(stdscr, top_y, 2, left_h, left_w, "ENIGMA FRONT PANEL", A_NORM)

        if ui.tm:
            _, lpos, mpos, rpos = decode_state(ui.tm.state)
        else:
            lpos, mpos, rpos = ui.settings.start_positions

        safe_add(stdscr, top_y+2, 4, "ROTOR WINDOWS", A_NORM | curses.A_BOLD)
        safe_add(stdscr, top_y+3, 4, f"L ({ui.settings.rotor_order[0]}): [{ALPH[lpos]}]   wheel: {rotor_wheel_strip(lpos)}"[:left_w-6], A_NORM)
        safe_add(stdscr, top_y+4, 4, f"M ({ui.settings.rotor_order[1]}): [{ALPH[mpos]}]   wheel: {rotor_wheel_strip(mpos)}"[:left_w-6], A_NORM)
        safe_add(stdscr, top_y+5, 4, f"R ({ui.settings.rotor_order[2]}): [{ALPH[rpos]}]   wheel: {rotor_wheel_strip(rpos)}"[:left_w-6], A_NORM)

        safe_add(stdscr, top_y+7, 4, "PLUGBOARD (Steckerbrett)", A_NORM | curses.A_BOLD)
        safe_add(stdscr, top_y+8, 4, f"Pairs: {pairs_str}"[:left_w-6], A_NORM)

        lamp_is_hot = (ui.last_out is not None and time.time() <= ui.lamp_flash_until)
        A_LAMP = (curses.color_pair(4) | curses.A_BOLD) if lamp_is_hot else curses.color_pair(4)

        # Keyboard/lampboard inside the left panel (staggered like real keyboard)
        draw_keyboard_and_lamps_in_box(
            stdscr,
            box_y=top_y, box_x=2, box_h=left_h, box_w=left_w,
            start_y=top_y+10,
            lit_in=ui.last_in,
            lit_out=ui.last_out,
            A_NORM=A_NORM,
            A_KEY=A_KEY,
            A_LAMP=A_LAMP
        )

        # Middle panels
        mid_x = 2 + left_w + 2
        draw_box(stdscr, top_y, mid_x, 10, mid_w, "TAPE (TM) + HEAD", A_NORM)

        if ui.tm:
            tape_line, head_line, left_idx = ui.tm.tape.snapshot(ui.tm.head, ui.tm.config.window)
            safe_add(stdscr, top_y+2, mid_x+2, f"State: {ui.tm.state}"[:mid_w-4], A_NORM)
            safe_add(stdscr, top_y+3, mid_x+2, f"Steps: {ui.tm.steps}   Head: {ui.tm.head}   Window starts: {left_idx}"[:mid_w-4], A_NORM)
            safe_add(stdscr, top_y+5, mid_x+2, tape_line[:mid_w-4], A_NORM)
            safe_add(stdscr, top_y+6, mid_x+2, head_line[:mid_w-4], A_TITLE)
        else:
            safe_add(stdscr, top_y+5, mid_x+2, "(not loaded)", A_STAT)

        out_y = top_y + 11
        draw_box(stdscr, out_y, mid_x, 7, mid_w, "OUTPUT", A_NORM)
        out = current_output()
        safe_add(stdscr, out_y+2, mid_x+2, out[:mid_w-4], A_TITLE)

        if ui.tm and ui.tm.halted():
            if ui.tm.state == "q_accept":
                safe_add(stdscr, out_y+4, mid_x+2, "HALT: ACCEPT", A_OK)
            elif ui.tm.state == "q_reject":
                safe_add(stdscr, out_y+4, mid_x+2, "HALT: REJECT", A_BAD)
            else:
                safe_add(stdscr, out_y+4, mid_x+2, f"HALT: {ui.tm.state}"[:mid_w-4], A_STAT)

        # Right panels
        right_x = mid_x + mid_w + 2
        wiring_lines = wiring_panel_lines(ui.settings)

        draw_box(stdscr, top_y, right_x, 10, right_w, "WIRING (scroll [ / ])", A_NORM)
        visible = 8
        start = ui.scroll_wiring
        for i in range(visible):
            idx = start + i
            if 0 <= idx < len(wiring_lines):
                safe_add(stdscr, top_y+2+i, right_x+2, wiring_lines[idx][:right_w-4], A_NORM)

        log_y = top_y + 11
        log_h = h - log_y - 3
        if log_h < 8:
            log_h = 8
        draw_box(stdscr, log_y, right_x, log_h, right_w, "SIGNAL PATH LOG (Up/Down)", A_NORM)

        lines = ui.path_log
        max_lines = log_h - 3
        if len(lines) > max_lines:
            ui.scroll_log = max(0, min(ui.scroll_log, len(lines) - max_lines))
        else:
            ui.scroll_log = 0

        view = lines[ui.scroll_log: ui.scroll_log + max_lines]
        for i, line in enumerate(view):
            safe_add(stdscr, log_y+2+i, right_x+2, line[:right_w-4], A_NORM)

        # Footer
        ctrl = "L load | S step | R run | E msg | P start | G ring | O rotors | B plug | D decrypt | C copy-out | +/− speed | Q quit"
        safe_add(stdscr, h-2, 2, ctrl[:w-4], A_NORM)
        safe_add(stdscr, h-1, 2, ui.status[:w-4], A_STAT)

        stdscr.refresh()

        # -----------------------------
        # INPUT
        # -----------------------------
        ch = stdscr.getch()
        if ch == -1:
            continue

        if ch in (ord('q'), ord('Q')):
            return

        if ch in (ord('r'), ord('R')):
            ui.running = not ui.running
            ui.status = "Running..." if ui.running else "Stopped."
            continue

        if ch in (ord('s'), ord('S')):
            if ui.tm:
                ui.tm.step()
                ui.status = "Stepped."
            continue

        if ch in (ord('+'), ord('=')):
            ui.speed_sps = min(250, ui.speed_sps + 5)
            continue

        if ch in (ord('-'), ord('_')):
            ui.speed_sps = max(1, ui.speed_sps - 5)
            continue

        if ch == ord('['):
            ui.scroll_wiring = max(0, ui.scroll_wiring - 1)
            continue

        if ch == ord(']'):
            ui.scroll_wiring = min(max(0, len(wiring_lines) - 8), ui.scroll_wiring + 1)
            continue

        if ch == curses.KEY_UP:
            ui.scroll_log = max(0, ui.scroll_log - 1)
            continue

        if ch == curses.KEY_DOWN:
            ui.scroll_log += 1
            continue

        if ch in (ord('l'), ord('L'), ord('x'), ord('X')):
            load_tm()
            ui.status = "Loaded/reset from current message and settings."
            continue

        if ch in (ord('e'), ord('E')):
            ui.running = False
            new_msg = prompt_line(stdscr, "Enter message (A–Z and space)", ui.message, A_STAT)
            ui.message = normalize(new_msg)
            ui.status = "Message updated. Press L to load."
            continue

        if ch in (ord('p'), ord('P')):
            ui.running = False
            raw = prompt_line(
                stdscr,
                "Start positions (Grundstellung) L M R",
                f"{ALPH[ui.settings.start_positions[0]]} {ALPH[ui.settings.start_positions[1]]} {ALPH[ui.settings.start_positions[2]]}",
                A_STAT
            )
            try:
                ui.settings.start_positions = parse_triplet_letters(raw)
                ui.status = "Start positions updated. Press L to load."
            except Exception as e:
                ui.status = f"Error: {e}"
            continue

        if ch in (ord('g'), ord('G')):
            ui.running = False
            raw = prompt_line(
                stdscr,
                "Ring settings (Ringstellung) L M R",
                f"{ALPH[ui.settings.ring_settings[0]]} {ALPH[ui.settings.ring_settings[1]]} {ALPH[ui.settings.ring_settings[2]]}",
                A_STAT
            )
            try:
                ui.settings.ring_settings = parse_triplet_letters(raw)
                ui.status = "Ring settings updated. Press L to load."
            except Exception as e:
                ui.status = f"Error: {e}"
            continue

        if ch in (ord('o'), ord('O')):
            ui.running = False
            raw = prompt_line(
                stdscr,
                "Rotor order (L M R) e.g., I II III",
                f"{ui.settings.rotor_order[0]} {ui.settings.rotor_order[1]} {ui.settings.rotor_order[2]}",
                A_STAT
            )
            try:
                ui.settings.rotor_order = parse_rotor_order(raw)
                ui.status = "Rotor order updated. Press L to load."
            except Exception as e:
                ui.status = f"Error: {e}"
            continue

        if ch in (ord('b'), ord('B')):
            ui.running = False
            default = " ".join([a+b for (a, b) in ui.settings.plugboard_pairs]) if ui.settings.plugboard_pairs else ""
            raw = prompt_line(stdscr, "Plugboard pairs (e.g., AQ EP MT) or blank", default, A_STAT)
            try:
                ui.settings.plugboard_pairs = parse_plugboard(raw)
                ui.status = "Plugboard updated. Press L to load."
            except Exception as e:
                ui.status = f"Error: {e}"
            continue

        if ch in (ord('c'), ord('C')):
            ui.message = current_output()
            ui.status = "Copied output into message. Press L to load and run again."
            continue

        if ch in (ord('d'), ord('D')):
            cipher = current_output()
            load_tm(cipher)
            ui.status = "Decrypt: loaded output as new message with SAME settings (Enigma is symmetric)."
            continue


def main():
    curses.wrapper(ui_loop)


if __name__ == "__main__":
    main()
