"""Scene-state and spatial-consistency engine.

Every AI shot can carry a typed "state" block: who is where on screen, which way they face, where the camera is,
which side of the action line it stands on, who holds which object, and an ordered list of SMALL events
(enter → walk_to → stop → place …). From that one description the engine
  1. simulates the world state shot by shot and rejects impossible or contradictory plans (this module),
  2. draws the layout control video that steers generation (layout_control.py),
  3. produces the expected tracks the generated clip is checked against (layout_control.expected_tracks →
     motion_check.py).
The plan is the single source of truth for generation AND validation.

Coordinates: normalised screen space, x from 0 (left edge) to 1 (right edge), y from 0 (top) to 1 (bottom).
Event times: fractions of the shot (0..1). Pure Python + pydantic v2; unit-tested in tests/test_scene_state.py.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

Box = tuple[float, float, float, float]          # x0, y0, x1, y1 (normalised)
Facing = Literal["left", "right", "camera", "away"]
LOCOMOTION = {"enter", "walk_to", "exit"}
MANIPULATION = {"take_out", "place", "pick_up", "hand_over", "scan", "open"}
VERSION = 1                                      # bump when the rules change (part of the validation cache key)


def cx(b: Box) -> float:
    return (b[0] + b[2]) / 2


def cy(b: Box) -> float:
    return (b[1] + b[3]) / 2


def near(a: Box, b: Box, gap: float = 0.08) -> bool:
    """Boxes touch or are within `gap` of each other (normalised)."""
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


class Camera(BaseModel):
    setup: str                                   # camera position id, e.g. "store_wide"
    axis: str                                    # action line in this location, e.g. "store_line"
    side: Literal["front", "back"] = "front"     # which side of the action line the camera is on (180° rule)
    size: Literal["wide", "medium", "close", "insert"] = "medium"
    move: Literal["static", "pan_left", "pan_right", "tilt_up", "tilt_down", "push_in", "pull_out",
                  "track_left", "track_right"] = "static"


class Placement(BaseModel):
    box: Box
    facing: Optional[Facing] = None

    @field_validator("box")
    @classmethod
    def _box_ok(cls, b):
        if not (b[0] < b[2] and b[1] < b[3]):
            raise ValueError(f"box {b} must be x0 < x1 and y0 < y1")
        if not all(-0.6 <= v <= 1.6 for v in b):           # slightly off-screen is allowed (entering/exiting)
            raise ValueError(f"box {b} is far outside the frame")
        return b


class Event(BaseModel):
    id: str
    type: Literal["enter", "walk_to", "stop", "take_out", "place", "pick_up", "hand_over", "scan", "look_at",
                  "talk", "sit", "stand", "open", "exit", "idle", "gesture"]
    actor: str
    obj: Optional[str] = None                    # object acted on (place / pick_up / hand_over / take_out / scan)
    target: Optional[str] = None                 # entity or anchor (counter, table…) the action is aimed at
    t: tuple[float, float]                       # start, end (fractions of the shot)
    to_box: Optional[Box] = None                 # where the actor ends (locomotion) or where the object ends (place)
    critical: bool = True                        # must be visibly completed; checked after generation
    describe: str = ""                           # wording for the prompt (generated if empty)

    @field_validator("t")
    @classmethod
    def _t_ok(cls, t):
        if not (0.0 <= t[0] < t[1] <= 1.0):
            raise ValueError(f"event time {t} must satisfy 0 <= start < end <= 1")
        return t


class ShotState(BaseModel):
    shot_id: str
    location: str
    camera: Camera
    start: dict[str, Placement]                  # visible entities at the first frame (characters, props, anchors)
    holders: dict[str, str] = Field(default_factory=dict)   # object -> who/what holds it at the first frame
    events: list[Event] = Field(default_factory=list)
    axis_change: Optional[str] = None            # reason, when the camera deliberately crosses the action line
    transition: Optional[str] = None             # "cut", "time_skip": allows positions (not ownership) to reset
    moved_offscreen: dict[str, str] = Field(default_factory=dict)   # object -> new holder, changed between shots
    offscreen_reason: Optional[str] = None       # required with moved_offscreen (e.g. "shipped to the returns centre")
    tail: float = 0.15                           # share of the shot kept AFTER the last critical event

    @model_validator(mode="after")
    def _events_ordered(self):
        last = 0.0
        for e in self.events:
            if e.t[0] < last - 1e-6 and e.type in LOCOMOTION | MANIPULATION:
                raise ValueError(f"{self.shot_id}: event {e.id} starts before the previous action ends")
            if e.type in LOCOMOTION | MANIPULATION:
                last = e.t[1]
        return self


class Entity(BaseModel):
    kind: Literal["character", "prop", "anchor"]
    name: str = ""
    small: bool = False                          # too small for reliable automatic detection (earbuds, cards)


class World(BaseModel):
    """What is true after a shot: positions are only comparable within the same camera axis and side."""
    location: Optional[str] = None
    axis: Optional[str] = None
    side: Optional[str] = None
    setup: Optional[str] = None
    boxes: dict[str, Box] = Field(default_factory=dict)
    facing: dict[str, Optional[str]] = Field(default_factory=dict)
    holders: dict[str, str] = Field(default_factory=dict)
    moving: dict[str, int] = Field(default_factory=dict)       # entity -> screen direction it was travelling (+1/-1)


class Issue(BaseModel):
    level: Literal["error", "warn"]
    rule: str
    shot: Optional[str]
    message: str
    fix: str = ""


def _issue(level, rule, shot, msg, fix=""):
    return Issue(level=level, rule=rule, shot=shot, message=msg, fix=fix)


def simulate(state: ShotState, before: World, entities: dict[str, Entity], max_actions=(1, 1)):
    """Apply one shot's events to the world. Returns (world after the shot, issues)."""
    iss = []
    sid = state.shot_id
    same_view = (before.location == state.location and before.axis == state.camera.axis and not state.transition)
    # ---- 180° rule: crossing the action line swaps left and right on screen
    if same_view and before.side and before.side != state.camera.side and not state.axis_change:
        iss.append(_issue("error", "X1", sid, f"the camera crosses the action line ({before.side} → "
                          f"{state.camera.side}) without a reason: left and right swap for the viewer",
                          "keep the camera on the same side, or set \"axis_change\" with the reason (e.g. a reveal)"))
    for name in list(state.start) + list(state.holders):
        if name not in entities:
            iss.append(_issue("error", "X0", sid, f"unknown entity '{name}'", "declare it in plan \"entities\""))
    # ---- ownership continues across cuts: objects only change hands through events (or a declared off-screen move)
    if state.moved_offscreen and not state.offscreen_reason:
        iss.append(_issue("error", "X2", sid, "\"moved_offscreen\" needs an \"offscreen_reason\"",
                          "say why the object changed place between shots (e.g. shipped)"))
    before = before.model_copy(update={"holders": {**before.holders, **state.moved_offscreen}})
    for obj, holder in state.holders.items():
        prev = before.holders.get(obj)
        if prev and prev != holder:
            iss.append(_issue("error", "X2", sid, f"'{obj}' was held by '{prev}' but this shot starts with "
                              f"'{holder}' holding it, and no event moved it", "add the hand-over/place event"))
    holders = dict(before.holders); holders.update(state.holders)
    # ---- left/right order and travel direction continue within the same view
    if same_view and state.camera.side == before.side:
        common = [e for e in state.start if e in before.boxes and entities.get(e, Entity(kind="prop")).kind != "anchor"]
        for a in common:
            for b in common:
                if a < b and (cx(before.boxes[a]) - cx(before.boxes[b])) * (cx(state.start[a].box) - cx(state.start[b].box)) < 0:
                    iss.append(_issue("error", "X3", sid, f"'{a}' and '{b}' swap left/right sides between shots",
                                      "keep their screen order, or change the axis on purpose (\"axis_change\")"))
    boxes = {k: v.box for k, v in state.start.items()}
    facing = {k: v.facing for k, v in state.start.items()}
    moving = {}
    n_loco = n_manip = 0
    for e in state.events:
        if e.actor not in boxes and e.type != "enter":
            iss.append(_issue("error", "E1", sid, f"event {e.id}: '{e.actor}' is not in the shot at that time",
                              "add them to \"start\" or begin with an \"enter\" event"))
            continue
        if e.type in LOCOMOTION:
            n_loco += 1
            if e.to_box is None:
                iss.append(_issue("error", "E2", sid, f"event {e.id} ({e.type}) needs \"to_box\" (where they end)"))
                continue
            if e.type == "enter":
                b0 = boxes.get(e.actor)
                if b0 is None:
                    iss.append(_issue("error", "E3", sid, f"event {e.id}: '{e.actor}' enters, so give their first, "
                                      "partly off-screen box in \"start\""))
                    continue
                if not (b0[0] < 0.05 or b0[2] > 0.95):
                    iss.append(_issue("error", "E3", sid, f"event {e.id}: '{e.actor}' enters but starts in the middle "
                                      "of the frame, not at an edge", "start the box at the left or right edge"))
            dx = cx(e.to_box) - cx(boxes[e.actor])
            if abs(dx) > 0.05:
                d = 1 if dx > 0 else -1
                moving[e.actor] = d
                want = "right" if d > 0 else "left"
                if facing.get(e.actor) in ("left", "right") and facing[e.actor] != want:
                    iss.append(_issue("error", "E4", sid, f"event {e.id}: '{e.actor}' faces {facing[e.actor]} but "
                                      f"walks {want} (walking backwards)", f"set facing \"{want}\""))
                facing[e.actor] = want
                prev_dir = before.moving.get(e.actor) if same_view and state.camera.side == before.side else None
                if prev_dir and prev_dir != d and not state.axis_change:
                    iss.append(_issue("error", "X4", sid, f"'{e.actor}' was moving {'right' if prev_dir > 0 else 'left'} "
                                      f"at the end of the last shot but moves {want} here (reversal)",
                                      "keep the direction, add a turn-around event, or set \"axis_change\""))
            boxes[e.actor] = e.to_box
        elif e.type == "stop":
            moving.pop(e.actor, None)
        elif e.type in MANIPULATION:
            n_manip += 1
            if not e.obj:
                iss.append(_issue("error", "E5", sid, f"event {e.id} ({e.type}) needs \"obj\""))
                continue
            holder = holders.get(e.obj)
            if e.actor in moving and e.type in ("place", "pick_up", "hand_over"):
                iss.append(_issue("error", "E6", sid, f"event {e.id}: '{e.actor}' is still walking while trying to "
                                  f"{e.type} '{e.obj}'", "add a \"stop\" event before it"))
            if e.type in ("place", "hand_over"):
                if holder != e.actor:
                    iss.append(_issue("error", "E7", sid, f"event {e.id}: '{e.actor}' {e.type}s '{e.obj}' but "
                                      f"'{holder}' holds it", "add the take_out/pick_up/hand_over that gives it to them"))
                tgt = boxes.get(e.target) if e.target else None
                if e.target and tgt is None:
                    iss.append(_issue("error", "E8", sid, f"event {e.id}: target '{e.target}' is not in the shot"))
                elif tgt and not near(boxes[e.actor], tgt):
                    iss.append(_issue("error", "E9", sid, f"event {e.id}: '{e.actor}' is not next to '{e.target}' "
                                      f"(can't {e.type} across the room)", "walk there first (walk_to + stop)"))
                holders[e.obj] = e.target if e.type == "place" else e.target
                if e.to_box:
                    boxes[e.obj] = e.to_box
            elif e.type in ("pick_up", "take_out"):
                if e.type == "pick_up":
                    src = holder
                    if src and src in boxes and not near(boxes[e.actor], boxes[src]):
                        iss.append(_issue("error", "E9", sid, f"event {e.id}: '{e.actor}' picks up '{e.obj}' from "
                                          f"'{src}' without being next to it"))
                elif holder not in (e.actor, None):
                    iss.append(_issue("error", "E7", sid, f"event {e.id}: '{e.actor}' takes out '{e.obj}' but "
                                      f"'{holder}' holds it"))
                holders[e.obj] = e.actor
            elif e.type == "scan":
                tgt = e.target or holder
                if tgt in boxes and not near(boxes[e.actor], boxes[tgt]):
                    iss.append(_issue("error", "E9", sid, f"event {e.id}: '{e.actor}' scans '{e.obj}' from too far away"))
    # ---- keep each AI clip simple: one movement + one hand action at most (models drop or merge more)
    if n_loco > max_actions[0] or n_manip > max_actions[1]:
        iss.append(_issue("error", "E10", sid, f"{n_loco} movements and {n_manip} hand actions in one clip "
                          f"(max {max_actions[0]} + {max_actions[1]}): the model will skip or merge some",
                          "split the shot into shorter shots, one action each"))
    crit = [e for e in state.events if e.critical]
    if crit and max(e.t[1] for e in crit) > 1 - state.tail + 1e-6:
        iss.append(_issue("error", "E11", sid, "the last required action ends too close to the cut "
                          f"(needs {state.tail:.0%} of the shot after it, so editing never cuts it off)",
                          "end the action earlier"))
    after = World(location=state.location, axis=state.camera.axis, side=state.camera.side, setup=state.camera.setup,
                  boxes=boxes, facing=facing, holders=holders, moving=moving)
    return after, iss


def check_sequence(states: list[ShotState], entities: dict[str, Entity], max_actions=(1, 1)):
    """Simulate a list of shots in order. Returns (worlds after each shot, all issues)."""
    world, worlds, issues = World(), [], []
    for st in states:
        if st.location != world.location or st.transition:
            # new place or time skip: positions reset, ownership carries over
            world = World(holders=world.holders)
        world, iss = simulate(st, world, entities, max_actions)
        worlds.append(world)
        issues += iss
    return worlds, issues


def parse_plan(plan: dict):
    """(states in shot order, entities, problems) from a plan's "entities" and per-shot "state" blocks."""
    problems = []
    ents = {}
    for k, v in (plan.get("scene_entities") or {}).items():
        try:
            ents[k] = Entity(**v)
        except Exception as e:                                      # pydantic ValidationError
            problems.append(_issue("error", "X0", None, f"entity '{k}': {e}"))
    states = []
    for s in plan.get("shots", []):
        if "state" not in s:
            continue
        try:
            states.append(ShotState(shot_id=s["shot_id"], **s["state"]))
        except Exception as e:
            problems.append(_issue("error", "X0", s["shot_id"], f"state block is invalid: {e}"))
    return states, ents, problems


def prompt_hints(state: ShotState, entities: dict[str, Entity]) -> str:
    """Plain-language spatial wording added to the prompt (the layout video does the real steering)."""
    def where(b):
        x = cx(b)
        return "on the left" if x < 0.4 else "on the right" if x > 0.6 else "in the centre"
    parts = []
    for k, p in state.start.items():
        nm = entities[k].name or k if k in entities else k
        if 0 <= cx(p.box) <= 1:
            parts.append(f"{nm} {where(p.box)} of the frame")
    for e in state.events:
        if e.describe:
            parts.append(e.describe)
    cam = {"static": "static camera", "push_in": "slow camera push in", "pull_out": "slow camera pull out"}.get(
        state.camera.move, state.camera.move.replace("_", " "))
    parts.append(f"{state.camera.size} shot, {cam}")
    return ", ".join(parts)
