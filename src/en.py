"""English variant (D-024): English-language fantasy with Japanese anime / light-novel themes.
Everything language-specific for English lives here: formats measured in words, the prompt format,
title templates, generator/labeler/judge instructions, rule filters, policy texts and metrics. The
genre taxonomy and the record schema are shared with the Japanese model (``format`` keeps the
canonical keys あらすじ/短編/続き; ``language`` is "en").
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from kitsune.data.filters import (
    _PII_RES,
    FilterOutcome,
    f_repetition,
    has_markdown,
)
from kitsune.taxonomy import GENRES, normalize_genres
if TYPE_CHECKING:
    from kitsune.data.generate import GenJob


def _genjob(**kw: Any) -> GenJob:
    from kitsune.data.generate import GenJob

    return GenJob(**kw)

# ----------------------------------------------------------------------------- taxonomy and formats
def _title_index(titles: list[str], threshold: float) -> Any:
    from kitsune.data.dedup import TitleIndex

    return TitleIndex(titles, threshold)

GENRE_NAME_EN: Final[dict[str, str]] = {
    "異世界転生": "Isekai",
    "悪役令嬢・転生": "Villainess",
    "魔王と勇者": "Demon Lord & Hero",
    "冒険者ギルド": "Adventurer Guild",
    "魔法学園": "Magic Academy",
    "魔法少女": "Magical Girl",
    "ダークファンタジー": "Dark Fantasy",
    "ハイファンタジー": "High Fantasy",
    "スローライフ": "Slow Life",
}
FORMAT_NAME_EN: Final[dict[str, str]] = {
    "あらすじ": "synopsis",
    "短編": "short story",
    "続き": "continuation",
}
FORMAT_FROM_EN: Final[dict[str, str]] = {v: k for k, v in FORMAT_NAME_EN.items()}

@dataclass(frozen=True)
class FormatSpecEN:
    target_min: int
    target_max: int
    filter_min: int
    filter_max: int
    max_new_tokens: int

FORMATS_EN: Final[dict[str, FormatSpecEN]] = {
    "あらすじ": FormatSpecEN(150, 350, 110, 420, 700),
    "短編": FormatSpecEN(600, 1100, 550, 1250, 1900),
    "続き": FormatSpecEN(300, 600, 220, 700, 1100),
}
PASSAGE_WORDS: Final = (120, 250)

def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))


def build_user_prompt_en(genres: list[str], title: str, fmt: str, passage: str | None = None) -> str:
    genres = normalize_genres(genres)
    lines = [
        f"Genres: {', '.join(GENRE_NAME_EN[g] for g in genres)}",
        f"Title: {title.strip()}",
        f"Format: {FORMAT_NAME_EN[fmt]}",
    ]
    if fmt == "続き":
        if not passage:
            raise ValueError("a passage is required for continuation")
        lines += ["Passage:", passage.strip()]
    elif passage:
        raise ValueError("a passage is only allowed for continuation")
    return "\n".join(lines)

_GENRE_FROM_EN: Final = {v.lower(): k for k, v in GENRE_NAME_EN.items()}


_J = ("Swordsman", "Mage", "Apothecary", "Blacksmith", "Alchemist", "Saint", "Knight", "Archer", "Summoner", "Enchanter", "Porter", "Healer", "Cook", "Bard", "Fortune Teller", "Dragon Rider", "Tamer", "Spirit Mage", "Barrier Mage", "Herbalist", "Librarian", "Scout")  # fmt: skip
_C = ("Exiled", "Banished", "Disowned", "Useless", "Kicked-Out", "Unskilled", "Weakest", "Forgotten", "Demoted", "Failed")  # fmt: skip
_P = ("the Frontier", "the Deep Forest", "a Corner of the Royal Capital", "the Demon Wilds", "an Old Lighthouse", "the Northern Snowlands", "a Seaside Village", "the Labyrinth City", "a Floating Island", "a Desert Oasis", "the Dragon Valley", "the Misty Canyon", "the Spirit Forest", "a Lakeside Town")  # fmt: skip
_G = ("Open an Apothecary", "Live a Slow Life", "Raise Monsters", "Rebuild My Domain", "Become the Strongest", "Run a Tiny Diner", "Live with a Dragon", "Recover Lost Magic", "Graduate from the Academy", "Save the World")  # fmt: skip
_T = ("Turns Out to Be the Strongest", "Is Adored for Some Reason", "Learns the Secret of the World", "Teams Up with an Old Rival", "Rewrites Fate", "Becomes the Demon Lord's Apprentice", "Finds a Hidden Dungeon", "Remembers a Past Life")  # fmt: skip
_R = ("a Legendary Sword", "a Village Well", "the Demon Castle's Gatekeeper", "an Ancient Dragon's Egg", "a Dungeon Treasure Chest", "the Saint's Pet Cat", "a Grimoire in the Library", "a Background Villager", "the Villainess's Maid", "a Scarecrow in an Herb Garden")  # fmt: skip
_S = ("Dropout", "Scholarship Student", "Transfer Student", "Zero-Mana Student", "Student Council President", "Library Committee Member")  # fmt: skip
_MG = ("Lumina", "Stella", "Aria", "Noel", "Sora", "Hikari", "Lily", "Kanon", "Yume", "Mikoto")  # fmt: skip
_MW = ("Is Late Again Today", "Forgot Her Transformation Chant", "Is About to Be Found Out", "Befriends Her Enemy", "Protects the Town Before Exams", "Faces Her Final Transformation")  # fmt: skip
_AD = ("Ashen", "Bloodstained", "Black", "Cursed", "Eclipsed", "Fallen", "Thorned", "Abyssal", "Withered")  # fmt: skip
_ND = ("Crown", "Saint", "Knights", "Witch", "Covenant", "Gravekeeper", "Bride", "Prince", "Bell")  # fmt: skip
_AH = ("Stargazing", "Silver", "Dawnlit", "Azure", "Ancient", "Emerald", "Twilight", "Thousand-Year", "Wind-Reading")  # fmt: skip
_NH = ("Kingdom", "Dragon", "Shrine Maiden", "Brigade", "Continent", "Temple", "Elf", "Hero", "Princess", "Pact", "Tower")  # fmt: skip
TEMPLATES_EN: Final[dict[str, tuple[str, ...]]] = {
    "異世界転生": (
        "I Was Reincarnated as {r}, So I'll {g}",
        "Reborn in Another World as a {j}, I {g}",
        "The {j} with Memories of a Past Life {t}",
        "Summoned to Another World, the {j} {t}",
    ),
    "悪役令嬢・転生": (
        "I Became the Villainess, So I'll {g} in {p}",
        "The {c} Duke's Daughter {t}",
        "The Condemned Villainess Moves to {p}",
        "The Villainess Becomes a {j}",
    ),
    "魔王と勇者": (
        "The Hero Who Defeated the Demon Lord Wants to {g}",
        "The Retired Demon Lord Wants to {g}",
        "The Demon Lord's Daughter and the {c} {j}",
        "The Hero Party's {j} {t}",
    ),
    "冒険者ギルド": (
        "The {c} {j} {t} at the Adventurer's Guild",
        "The Guild Receptionist {t}",
        "An F-Rank {j} {t}",
        "At the Guild in {p}, a {j} Wants to {g}",
    ),
    "魔法学園": (
        "The {s} of the Magic Academy {t}",
        "A {c} {j} Enrolls at the Magic Academy",
        "The Magic Academy's {s} Wants to {g}",
        "Secrets of the Academy in {p}",
    ),
    "魔法少女": (
        "Magical Girl {mg} {mw}",
        "The Magical Girl of {p} {mw}",
        "Magical Girl {mg} and the {c} Mascot",
        "The Former Magical Girl Wants to {g}",
    ),
    "ダークファンタジー": (
        "The {ad} {nd}",
        "The {ad} {nd} and the {ad} {nd}",
        "The {c} {j} and the {ad} {nd}",
        "The {ad} {nd} Sleeps in {p}",
    ),
    "ハイファンタジー": (
        "The {ah} {nh}",
        "The {ah} {nh} and the {ah} {nh}",
        "The {j} of the {ah} {nh}",
        "Chronicles of {p}: The {ah} {nh}",
    ),
    "スローライフ": (
        "A {c} {j} Wants a Quiet Life in {p}",
        "A {j}'s Slow Life in {p}",
        "The Little Shop in {p}",
        "In {p}, I'll {g}",
    ),
}

# Long English titles from one template share many character bigrams, so the Japanese threshold (0.6)
EN_TITLE_NEAR: Final = 0.85

def make_title_en(genre: str, rng: random.Random) -> str:
    tpl = rng.choice(TEMPLATES_EN[genre])
    slots = {
        "j": rng.choice(_J),
        "c": rng.choice(_C),
        "p": rng.choice(_P),
        "g": rng.choice(_G),
        "t": rng.choice(_T),
        "r": rng.choice(_R),
        "s": rng.choice(_S),
        "mg": rng.choice(_MG),
        "mw": rng.choice(_MW),
    }
    out = tpl
    for key, pool in (("ad", _AD), ("nd", _ND), ("ah", _AH), ("nh", _NH)):
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(pool), 1)
    return out.format(**slots)

def build_train_prompts_en(
    n: int,
    test_titles: list[str],
    seed: int,
    fmt_weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
    strict: bool = True,
) -> list[dict]:
    """``n`` templated training prompts (each title used at most twice, none near a test title).

    ``strict=False`` returns fewer than ``n`` prompts instead of raising when the template space runs out.
    """
    rng = random.Random(seed)
    idx = _title_index(test_titles, EN_TITLE_NEAR)
    counts: dict[str, int] = {}
    out: list[dict] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 200:
            if not strict:
                break
            raise RuntimeError("English title space exhausted")
        g = GENRES[len(out) % len(GENRES)]
        t = make_title_en(g, rng)
        if counts.get(t, 0) >= 2 or idx.is_near(t):
            continue
        counts[t] = counts.get(t, 0) + 1
        out.append(
            {
                "id": f"en-train-{len(out):06d}",
                "genres": _genres(g, rng),
                "title": t,
                "format": rng.choices(["あらすじ", "短編", "続き"], fmt_weights)[0],
                "title_source": "template",
            }
        )
    return out

# (generator default-name collapse). Each prompt therefore suggests a protagonist name from a varied pool.
NAMES_F_EN: Final = ("Mina", "Rin", "Sayo", "Yuna", "Hana", "Aoi", "Iris", "Mira", "Tessa", "Nadia", "Sora", "Ilse", "Kiri", "Noa", "Liesel", "Maren", "Chiyo", "Emi", "Fiora", "Greta", "Hazel", "Isolde", "Juno", "Kaya", "Lena", "Mei", "Nell", "Orla", "Pia", "Rika", "Saki", "Tove", "Vera", "Wren", "Yui", "Zara", "Anya", "Beatrix", "Cora", "Dalia")  # fmt: skip
NAMES_M_EN: Final = ("Leon", "Haru", "Soren", "Taro", "Kenji", "Arlo", "Bram", "Cyril", "Dario", "Emil", "Finn", "Gideon", "Hugo", "Ivo", "Jiro", "Kai", "Lucan", "Matteo", "Nils", "Oren", "Piet", "Rowan", "Sho", "Tomas", "Ulric", "Viktor", "Wade", "Yusuke", "Zeno", "Akira", "Daichi", "Felix", "Goro", "Hiro", "Isaac", "Jonas", "Koji", "Lars", "Milo", "Otto")  # fmt: skip
OVERUSED_NAMES_EN: Final = (
    "Elara",
    "Kaelen",
    "Kael",
    "Aethelgard",
    "Oakhaven",
    "Thorne",
    "Seraphina",
    "Valerius",
    "Lyra",
)
_FEMALE_PROT = {"a girl", "a woman"}
_MALE_PROT = {"a boy", "a young man", "an old knight"}

def titles_job_en(genre: str, rng: random.Random, i: int) -> GenJob:
    raise NotImplementedError
