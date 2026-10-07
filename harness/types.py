#!/usr/bin/env python3
"""Per-type colours and conflict resolution for multi-type images.

A single-degradation label cannot describe an image that is both dark and rainy,
so this repository lets one image carry several types. When it does, the task
block of EVERY type present is injected in full, and the one axis on which they
actually disagree is resolved here.

The axis is colour. Two of the six types necessarily change colour and
brightness as part of doing their job:

* haze - clearing an airlight veil is a change of colour and contrast
* lowlight - lifting a dark scene raises brightness and saturation

The other three (rain, snow, mix) lock colour. Between themselves they
never conflict - locking intersected with locking is still locking. So the only
case needing a ruling is "a colour-changing type is present alongside a
colour-locking one", and there the colour-changing side wins: the locking clause
was written for an image without that degradation, and obeying it would make the
restoration impossible.

Because the usual guard is relaxed in exactly that case, the ruling has to say
what still holds - no gratuitous recolouring, no hue-category flips, no
colourising a monochrome image - and it has to say it in a form the composer can
act on ("state this in the instruction") and, separately, in a form the verifier can
score ("count this as a defect"). Those two audiences are why the colour ruling
exists twice, in two voices.
"""
from __future__ import annotations

from .utils import types_label

# The two types whose restoration necessarily changes colour and brightness.
COLOUR_CHANGING_TYPES = frozenset({"haze", "lowlight"})

_WHY = {
    "haze": "clearing an airlight veil",
    "lowlight": "lifting a dark scene",
}


def _colour_clause(types: list[str]) -> str:
    """Colour ruling addressed to the composer."""
    changing = [t for t in types if t in COLOUR_CHANGING_TYPES]
    locking = [t for t in types if t not in COLOUR_CHANGING_TYPES]

    if not changing:
        return (
            "  COLOUR: none of the types present licenses a colour change. Global "
            "colour, tone and white balance stay exactly as they are - no hue shift, "
            "no saturation shift, no recolouring, no white-balance change.")

    names = " and ".join(changing)
    if not locking:
        return (
            f"  COLOUR: {names} - colour and brightness necessarily change as a "
            "consequence of this restoration. Do not impose a 'colour must stay "
            "identical' requirement. Still avoid stylised recolouring: the target is "
            "how the scene would look under better conditions, not a graded look.\n"
            "     Every surface keeps its own true colour - what lifts is the grey or "
            "dark film sitting ON TOP of it. A red banner becomes a cleaner red, not a "
            "different hue; foliage becomes a truer green, not pink. State this in the "
            "instruction: every object keeps its own real-world colour, no region is "
            "repainted.")

    reason = " and ".join(_WHY[t] for t in changing)
    return (
        f"  COLOUR - THIS IS THE ONE REAL CONFLICT, AND {names.upper()} WINS IT.\n"
        f"     {names} cannot be performed without changing colour and brightness: "
        f"{reason} shifts them by physical necessity. The other type(s) present "
        f"({', '.join(locking)}) carry a 'colour unchanged' clause written for images "
        "WITHOUT that degradation. Here it does not apply.\n"
        "     So: allow the colour and brightness change that the restoration itself "
        "requires, and no more. What still holds from the locking type(s) is the ban "
        "on GRATUITOUS recolouring - no stylisation, no hue-category flips (red stays "
        "red, yellow lights stay yellow), no white-balance 'correction' done for its "
        "own sake, and no colourising a monochrome image.\n"
        "     WHAT 'ALLOWED TO CHANGE' MEANS, PRECISELY: every surface keeps its own "
        "true colour - the colour it would show under clear light. What lifts is the "
        "grey/dark film sitting ON TOP of that colour. A red banner becomes a cleaner "
        "red, not a different hue; foliage becomes a truer green, not pink; a concrete "
        "overpass stays grey concrete. Because the usual 'colour unchanged' guard is "
        "relaxed here, your instruction MUST say this explicitly - write that every "
        "object keeps its own real-world colour and that no region is to be repainted. "
        "That sentence is what confines the change to the film of degradation instead "
        "of the surfaces under it.")


def _colour_clause_verify(types: list[str]) -> str:
    """Colour ruling addressed to the verifier.

    Same physics as ``_colour_clause``; the difference is what the reader is asked
    to do with it. Injecting the composer version into the verifier's system prompt
    would instruct the verifier to compose a restoration prompt.
    """
    changing = [t for t in types if t in COLOUR_CHANGING_TYPES]
    locking = [t for t in types if t not in COLOUR_CHANGING_TYPES]

    if not changing:
        return (
            "  COLOUR: none of the types present licenses a colour change. Count as a "
            "defect any global hue shift, saturation shift, recolouring or "
            "white-balance change between input and result.")

    names = " and ".join(changing)
    expected = (
        "     A global colour and brightness change is therefore EXPECTED here. Do NOT "
        "record it as a defect on its own, and do not fail the result for 'the colours "
        "changed'.\n"
        "     What you DO still count as a defect: a hue-category flip (a red banner "
        "coming back a different hue, yellow lights turning white), stylised or graded "
        "colour, a whole region repainted a colour it never had, and colourising a "
        "monochrome image. The rule is that every surface keeps its own true colour - "
        "what lifts is the grey or dark film sitting ON TOP of it.")

    if not locking:
        return (f"  COLOUR: {names} - colour and brightness change as a physical "
                f"consequence of this restoration.\n" + expected)

    reason = " and ".join(_WHY[t] for t in changing)
    return (
        f"  COLOUR - THE TYPE BLOCKS CONFLICT HERE, AND {names.upper()} WINS.\n"
        f"     {names} cannot be performed without changing colour and brightness: "
        f"{reason} shifts them by physical necessity. The other type block(s) present "
        f"({', '.join(locking)}) carry a 'colour unchanged' item written for images "
        "WITHOUT that degradation. **Mark that item 'na' for this image** - do not "
        "fail the result against it.\n" + expected)


_RULES_HEAD = """
WHERE THESE TYPES CONFLICT

The blocks above were each written for a single-degradation image. Read together they
overlap far more than they disagree - the requirements simply add up. There is exactly
ONE axis on which they genuinely conflict, and it is settled for you below.
"""

_RULES_TAIL = """
  OTHER AXES DO NOT CONFLICT. Noise, blur, artifacts, weather occlusion and physical
  damage are treated independently and additively - doing one does not undo another.
  Where two blocks ask for the same thing in different words, say it once.

  PROTECTIONS ALWAYS SURVIVE. Settled snow is scene content; night stays night; an old
  photograph is never colourised or renovated; text and faces keep their identity;
  nothing is added, removed or reshaped. No "require" from one block overrides a
  "must not" from another on these.

  TREAT EVERY DEGRADATION THAT IS ACTUALLY PRESENT. Multiple types does not mean
  picking the dominant one and ignoring the rest. If the image is both dark and rainy,
  the instruction must address the darkness AND the rain.

  WHERE A BLOCK DESCRIBES SOMETHING THIS IMAGE DOES NOT HAVE, SKIP IT SILENTLY.
  A type label is a prior, not a guarantee. Do not instruct the editor to remove rain
  streaks you cannot see.

Do not add a discussion of the type system itself to the instruction.
"""

_RULES_TAIL_JUDGE = """
  OTHER AXES DO NOT CONFLICT. Noise, blur, artifacts, weather occlusion and physical
  damage are judged independently and additively - doing one does not excuse skipping
  another. Where two blocks ask for the same thing in different words, judge it once.

  PROTECTIONS ALWAYS SURVIVE. Settled snow is scene content; night stays night; an old
  photograph is never colourised or renovated; text and faces keep their identity;
  nothing is added, removed or reshaped. No "require" from one block excuses violating
  a "must not" from another on these - a violation is still a defect.

  EVERY DEGRADATION THAT WAS ACTUALLY PRESENT MUST HAVE BEEN TREATED. Multiple types
  does not mean the system may pick the dominant one and ignore the rest. If the input
  was both dark and rainy, a result that fixed only the darkness is under-doing.

  WHERE A BLOCK DESCRIBES SOMETHING THIS IMAGE DOES NOT HAVE, MARK ITS ITEMS "na".
  A type label is a prior, not a guarantee. Do not fail a result for failing to remove
  rain streaks that were never in the input.
"""


def conflict_rules(types: list[str], voice: str = "compose") -> str:
    """Build the conflict-resolution section for this image's type combination.

    ``voice="compose"`` feeds the composer (stage 3); ``"verify"`` feeds the
    verifier (stage 5). The physics is identical, only the addressee differs.
    """
    if voice == "verify":
        return (_RULES_HEAD.strip() + "\n\n" + _colour_clause_judge(types) + "\n"
                + _RULES_TAIL_JUDGE.rstrip())
    return (_RULES_HEAD.strip() + "\n\n" + _colour_clause(types) + "\n"
            + _RULES_TAIL.rstrip())


def build_system(template: str, types: list[str], blocks: dict[str, str],
                 voice: str = "compose") -> str:
    """Fill a prompt template for a (possibly multi-type) image.

    ``template`` carries ``{{TASK_TYPE}}`` and ``{{TASK_BLOCK}}``. ``blocks`` maps
    type to task block, as read by ``load_blocks``.

    An unknown type raises rather than falling back to ``mix``: a fallback would
    turn a one-character annotation slip into a whole run measured against the
    wrong standard, and nothing in the output would look wrong.
    """
    if not types:
        raise ValueError("build_system: no types given")

    unknown = [t for t in types if t not in blocks]
    if unknown:
        raise ValueError(
            f"unknown task type(s) {unknown}; available: {sorted(blocks)}. "
            f"Multiple types are separated with '|', e.g. "
            f"rain|lowlight.")

    colour = _colour_clause_verify if voice == "verify" else _colour_clause

    if len(types) == 1:
        # The colour ruling is injected for single-type images too. The
        # restoration spec deliberately drops the old "do not change brightness"
        # and "do not correct colour casts" items, because they contradict haze
        # and lowlight outright, and haze is single-typed. Without this clause a
        # single-type haze image would carry no colour guard at all.
        # Only the colour section is added - there is no conflict to resolve.
        task_block = blocks[types[0]] + "\n\n" + colour(types).strip()
    else:
        n = len(types)
        parts = [
            f"This image carries {n} degradation types at once. The specification "
            f"block for EACH is given below, in full. All of them apply.",
            "",
        ]
        for i, t in enumerate(types, start=1):
            parts.append(f"--------- [Type {i} of {n}: {t}] ---------")
            parts.append(blocks[t])
            parts.append("")
        parts.append(conflict_rules(types, voice=voice).strip())
        task_block = "\n".join(parts)

    return (template.replace("{{TASK_TYPE}}", types_label(types))
                    .replace("{{TASK_BLOCK}}", task_block))
