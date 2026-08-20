"""The Hinglish / Indian-English stress set. Hand-authored, hand-labelled.

Why this file exists even though the corpus contains Hindi. Reading the dataset
card on Day 1 turned up ``hin``, ``ben`` and ``mar`` among the 23 languages, which
was better news than the plan assumed. But *Hindi* and *Hinglish* are not the
same thing, and the gap is exactly where a voice agent for Indian users breaks:

* the corpus has monolingual Hindi, not **code-switched** speech;
* it has no **Indian-English** accent slice;
* and it has no logistics-domain vocabulary, which is where the shorthand that
  sounds like a sentence ending actually lives ("POD bhej diya", "GR number").

So the corpus gives a real ``hin`` evaluation slice — reported separately — and
this file gives the code-switching and domain coverage it lacks.

**The labelling contract.** ``endpoint=False`` means a detector that fires here
has produced a false interruption. ``endpoint=True`` means the utterance is
genuinely complete. The hard cases — and the reason this set is worth building —
are the ``False`` rows that *sound* finished: a trailing filler, a hesitation
before a number, a code-switch boundary where the Hindi clause has closed but the
English one has not started.

Vocabulary and phrasing are carried over from the voice-agent work's driver
phrase set, so the domain shorthand is real rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phrase:
    text: str
    endpoint: bool
    category: str
    note: str = ""
    # Bulbul is script-driven, so Devanagari and Latin text both voice correctly
    # under hi-IN. Latin-script rows are the code-switching ones.
    language_code: str = "hi-IN"

    @property
    def label(self) -> int:
        return int(self.endpoint)


# --------------------------------------------------------------------------- #
# NEGATIVES — a detector that fires on these interrupts a real user
# --------------------------------------------------------------------------- #
END_FILLER = [
    Phrase("Haan woh load ka matlab", False, "end_filler",
           "trails on 'matlab' — the single most common Hinglish hesitation"),
    Phrase("Pickup ka time hai umm", False, "end_filler", "bare filler at the end"),
    Phrase("Gaadi ka number hai actually", False, "end_filler",
           "'actually' as a Hinglish discourse particle, not a sentence end"),
    Phrase("Delivery address wala jo hai na woh", False, "end_filler",
           "'woh' dangling — speaker is still retrieving"),
    Phrase("Toh main keh raha tha ki", False, "end_filler",
           "subordinating 'ki' demands a clause that has not arrived"),
    Phrase("Bhaiya ek minute yeh dekh ke bataata hoon ki", False, "end_filler"),
    Phrase("Us truck ka jo driver hai uska naam", False, "end_filler",
           "possessive chain left open"),
    Phrase("Load kitna hai woh mujhe", False, "end_filler"),
    Phrase("You know woh", False, "end_filler", "English filler inside Hindi frame"),
    Phrase("Sir woh baat aisi hai ki", False, "end_filler"),
]

MID_HESITATION = [
    Phrase("Nashik se umm Pune load bhejna hai", False, "mid_hesitation",
           "filler mid-sentence; the pause is not a boundary"),
    Phrase("Vehicle number MH matlab MH12 AB 1234 hai", False, "mid_hesitation",
           "self-correction mid-number — a long pause that must not fire"),
    Phrase("Pincode chaar ek ek umm shunya shunya ek", False, "mid_hesitation",
           "digit-by-digit with a hesitation, the classic false-fire trap"),
    Phrase("Halt charges lagenge kya agar factory pe", False, "mid_hesitation",
           "conditional clause incomplete"),
    Phrase("Advance payment kab tak", False, "mid_hesitation", "question truncated"),
    Phrase("Maine POD bhej diya hai WhatsApp pe lekin", False, "mid_hesitation",
           "'lekin' promises a contrast that has not come"),
    Phrase("Route change ho gaya hai ab", False, "mid_hesitation"),
    Phrase("Container milega ya", False, "mid_hesitation", "disjunction left hanging"),
]

CODE_SWITCH_OPEN = [
    Phrase("The load is ready but gaadi", False, "code_switch",
           "switches to Hindi and stops mid-noun-phrase"),
    Phrase("Driver ko bola hai that he should", False, "code_switch",
           "switches to English and stops mid-clause"),
    Phrase("Total weight is around eighteen tonne aur", False, "code_switch",
           "'aur' opens a list that never continues"),
    Phrase("Delivery ka time confirm karke", False, "code_switch",
           "conjunctive participle: grammatically requires a main verb"),
    Phrase("I will send the GR number jaise hi", False, "code_switch",
           "'jaise hi' opens a temporal clause"),
]

# --------------------------------------------------------------------------- #
# POSITIVES — genuinely complete utterances
# --------------------------------------------------------------------------- #
COMPLETE_HINGLISH = [
    Phrase("Nashik se Pune load bhejna hai kal subah", True, "complete"),
    Phrase("Mera vehicle number MH12 AB 1234 hai", True, "complete"),
    Phrase("Load atharah tonne ka hai", True, "complete"),
    Phrase("Maine POD WhatsApp pe bhej diya hai", True, "complete"),
    Phrase("Pincode chaar ek ek shunya shunya ek hai", True, "complete",
           "same digits as the negative above, but resolved"),
    Phrase("Traffic bahut hai, ETA do ghante late hoga", True, "complete"),
    Phrase("Haan bhaiya theek hai", True, "complete", "short but complete"),
    Phrase("Indent confirm kar dijiye", True, "complete"),
    Phrase("Advance payment loading ke baad chahiye", True, "complete"),
    Phrase("Open truck chalega", True, "complete"),
    Phrase("Halt charges lagenge kya", True, "complete", "complete question"),
    Phrase("Theek hai", True, "complete", "two words — the shortest real endpoint"),
]

COMPLETE_ENGLISH_IN = [
    Phrase("I need a truck from Nashik to Pune tomorrow morning", True, "complete_en",
           "Indian-English accent slice", "en-IN"),
    Phrase("The load is eighteen tonnes and it is ready for pickup", True, "complete_en",
           language_code="en-IN"),
    Phrase("Please confirm the indent before six o'clock", True, "complete_en",
           language_code="en-IN"),
    Phrase("What is the GR number for this trip", True, "complete_en",
           language_code="en-IN"),
]

COMPLETE_DEVANAGARI = [
    Phrase("नाशिक से पुणे लोड भेजना है, कल सुबह गाड़ी चाहिए", True, "complete_dev",
           "Devanagari script — matches the corpus's own hin rows"),
    Phrase("गाड़ी का नंबर एम एच बारह ए बी बारह चौंतीस है", True, "complete_dev"),
    Phrase("मुझे कल सुबह ट्रक चाहिए", True, "complete_dev"),
]

INCOMPLETE_DEVANAGARI = [
    Phrase("नाशिक से पुणे लोड भेजना है लेकिन", False, "end_filler_dev"),
    Phrase("गाड़ी का नंबर मतलब", False, "end_filler_dev"),
]

# --------------------------------------------------------------------------- #
ALL_PHRASES: list[Phrase] = [
    *END_FILLER,
    *MID_HESITATION,
    *CODE_SWITCH_OPEN,
    *COMPLETE_HINGLISH,
    *COMPLETE_ENGLISH_IN,
    *COMPLETE_DEVANAGARI,
    *INCOMPLETE_DEVANAGARI,
]

# Bulbul speakers to render each phrase with. Multiple voices per phrase is what
# stops the set from measuring one voice's prosody instead of the phenomenon —
# with a single speaker, a model could score well by memorising that voice. Two
# male and two female, so the set is not a measurement of one pitch range.
#
# The roster is **model-specific**: bulbul:v3 rejects the v2-era names (anushka,
# abhilash, karun, hitesh) that the voice-agent work used. If a call fails with
# "not compatible with model", the error body lists the current valid names —
# BULBUL_V3_SPEAKERS below is that list as of Aug 2026.
SPEAKERS: tuple[str, ...] = ("shubh", "ritu", "aditya", "kavya")

BULBUL_V3_SPEAKERS: tuple[str, ...] = (
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan",
    "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "manan", "sumit", "roopa", "kabir", "aayan", "shubh", "advait", "anand",
    "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
)


def summary() -> str:
    from collections import Counter

    cats = Counter(p.category for p in ALL_PHRASES)
    pos = sum(p.endpoint for p in ALL_PHRASES)
    return (
        f"{len(ALL_PHRASES)} phrases "
        f"({pos} endpoint / {len(ALL_PHRASES) - pos} not-endpoint) "
        f"x {len(SPEAKERS)} speakers = {len(ALL_PHRASES) * len(SPEAKERS)} clips\n"
        + "\n".join(f"  {k:<18s} {v:>3d}" for k, v in sorted(cats.items()))
    )


if __name__ == "__main__":
    print(summary())
