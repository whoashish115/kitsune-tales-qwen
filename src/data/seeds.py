"""Seed prompts: light-novel-style titles, genre combinations, formats, and the frozen test set.

Titles come from hand-written templates and word lists (authored for this project,
Apache-2.0). Everything is driven by an explicit ``random.Random(seed)``, so the same seed
always yields the same prompts. The held-out test set is built first, written to
``data/test_prompts.jsonl`` and hashed; training titles are then drawn with a different seed
and filtered against it with exact and near-duplicate matching (``dedup.title_is_near``).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Final

from kitsune.data.dedup import TitleIndex, title_is_near
from kitsune.data.filters import f_prompt_safety
from kitsune.taxonomy import GENRES

# ----------------------------------------------------------------------------- word lists

JOBS: Final = (
    "剣士", "魔術師", "薬師", "鍛冶師", "錬金術師", "聖女", "騎士", "弓使い", "召喚士", "付与術師", "荷物持ち",
    "治癒士", "料理人", "吟遊詩人", "占い師", "竜騎士", "盾役", "精霊術師", "魔道具職人", "テイマー", "結界師",
    "書庫番", "庭師", "薬草摘み", "見習い魔女", "魔法剣士", "斥候", "調香師",
)  # fmt: skip
CASTOUT: Final = (
    "追放された", "婚約破棄された", "パーティを外された", "無能と呼ばれた", "役立たずと言われた", "落第した",
    "見捨てられた", "左遷された", "勘当された", "才能なしと判定された", "外れスキル持ちの", "最弱と笑われた",
)  # fmt: skip
PLACES: Final = (
    "辺境", "森の奥", "王都の片隅", "魔境", "古い灯台", "北の雪国", "海辺の村", "迷宮都市", "浮遊島",
    "砂漠のオアシス", "湖畔の町", "竜の谷", "霧の峡谷", "精霊の森", "国境の砦", "星降る丘",
)  # fmt: skip
GOALS: Final = (
    "スローライフを送る", "最強になる", "薬屋を開く", "魔物を育てる", "領地を立て直す", "世界を救う",
    "のんびり暮らす", "真の力に目覚める", "伝説の鍛冶師になる", "小さな食堂を営む", "竜と暮らす",
    "学園を卒業する", "失われた魔法を取り戻す", "王国一の冒険者になる", "誰かの役に立つ",
)  # fmt: skip
REBORN_AS: Final = (
    "伝説の剣", "村の井戸", "魔王城の門番", "古代竜の卵", "薬草園の案山子", "ダンジョンの宝箱", "聖女の飼い猫",
    "図書館の魔導書", "勇者の盾", "森の精霊", "辺境伯の三男", "ギルドの看板娘", "魔法学園の用務員",
    "悪役令嬢の侍女", "魔王の娘", "モブ村人", "宿屋の息子", "雪山の番人",
)  # fmt: skip
ADJ_DARK: Final = (
    "灰燼の", "血濡れの", "黒き", "呪われた", "終焉の", "月蝕の", "亡国の", "骸の", "絶望の", "朽ちた", "茨の", "奈落の",
)  # fmt: skip
NOUN_DARK: Final = (
    "王冠", "聖女", "騎士団", "魔女", "契約", "墓守", "剣", "祈り", "花嫁", "王子", "記憶", "鐘", "書架", "誓約",
)  # fmt: skip
ADJ_HIGH: Final = (
    "星詠みの", "銀の", "暁の", "蒼穹の", "古き", "風読みの", "翠玉の", "光輝の", "黄昏の", "千年の", "白銀の", "天空の",
)  # fmt: skip
NOUN_HIGH: Final = (
    "王国", "竜", "巫女", "旅団", "大陸", "神殿", "エルフ", "英雄", "王女", "盟約", "秘宝", "叙事詩", "塔", "旅路",
)  # fmt: skip
TWISTS: Final = (
    "実は最強だった", "なぜか懐かれる", "世界の秘密を知る", "宿敵と手を組む", "前世の記憶を取り戻す",
    "運命を書き換える", "ひとりで国を守る", "魔王に弟子入りする", "勇者に求婚される", "隠しダンジョンを見つける",
)  # fmt: skip
MAGICAL_GIRL_NAMES: Final = (
    "ルミナ", "ステラ", "ミルフィ", "アリア", "ノエル", "ソラ", "ヒカリ", "リリィ", "カノン", "ユメ", "ミコト", "フレア",
)  # fmt: skip
MAGICAL_GIRL_WORRIES: Final = (
    "今日も遅刻する", "変身の呪文を忘れる", "マスコットと喧嘩中", "正体がばれそう", "引退を考えている",
    "敵の少女と友達になる", "テスト前に街を守る", "最後の変身に挑む",
)  # fmt: skip
SCHOOL_ROLES: Final = (
    "落ちこぼれ", "特待生", "転入生", "劣等生", "図書委員", "生徒会長", "魔力ゼロの生徒", "教師見習い", "寮長",
)  # fmt: skip

# Primary-genre templates. {j}=job {c}=castout {p}=place {g}=goal {r}=reborn-as {t}=twist ...
TEMPLATES: Final[dict[str, tuple[str, ...]]] = {
    "異世界転生": (
        "転生したら{r}だったので、{g}",
        "異世界に転生した{j}は、{p}で{g}",
        "前世の記憶を持つ{j}、{p}で{t}",
        "{r}に転生した俺が{t}件",
        "召喚された{j}は、{p}で{g}",
    ),
    "悪役令嬢・転生": (
        "悪役令嬢に転生したので、{p}で{g}",
        "{c}令嬢は、{p}で{g}",
        "断罪された令嬢ですが、{t}ようです",
        "悪役令嬢の私が{j}として{g}",
        "婚約破棄された公爵令嬢、{p}で{t}",
    ),
    "魔王と勇者": (
        "魔王を倒した勇者は、{p}で{g}",
        "{c}勇者と{p}の魔王",
        "勇者パーティの{j}は、魔王に{t}",
        "魔王の娘と{c}{j}",
        "引退した魔王は{p}で{g}",
    ),
    "冒険者ギルド": (
        "{c}{j}、冒険者ギルドで{t}",
        "冒険者ギルドの受付嬢は{t}",
        "{p}のギルドで{j}として{g}",
        "Fランク冒険者の{j}は{t}",
        "{c}{j}は、ギルドの依頼で{g}",
    ),
    "魔法学園": (
        "魔法学園の{s}は{t}",
        "{c}{j}、魔法学園で{g}",
        "{p}の魔法学院と{s}の秘密",
        "魔法学園の{s}ですが、{t}ようです",
        "{s}の私が魔法学園で{g}",
    ),
    "魔法少女": (
        "魔法少女{mg}は{mw}",
        "{p}の魔法少女は{mw}",
        "魔法少女{mg}と{c}マスコット",
        "元魔法少女の{j}は{g}",
        "魔法少女{mg}、{t}",
    ),
    "ダークファンタジー": (
        "{ad}{nd}",
        "{ad}{nd}と{ad}{nd}",
        "{c}{j}と{ad}{nd}",
        "{ad}{nd}は{p}で眠る",
        "{ad}{nd}、あるいは{t}物語",
    ),
    "ハイファンタジー": (
        "{ah}{nh}",
        "{ah}{nh}と{ah}{nh}",
        "{ah}{nh}の{j}",
        "{p}戦記 ―{ah}{nh}―",
        "{ah}{nh}と{c}{j}",
    ),
    "スローライフ": (
        "{p}で{g}",
        "{c}{j}は、{p}でのんびり暮らしたい",
        "{p}で始める{j}のスローライフ",
        "{c}{j}ですが、{p}の暮らしが幸せすぎる",
        "{p}の小さな店と{j}の日々",
    ),
}

FORMAT_WEIGHTS: Final[dict[str, float]] = {"あらすじ": 0.25, "短編": 0.50, "続き": 0.25}


def make_title(genre: str, rng: random.Random) -> str:
    """Fill a random template of ``genre`` from the word lists."""
    tpl = rng.choice(TEMPLATES[genre])
    slots = {
        "j": rng.choice(JOBS),
        "c": rng.choice(CASTOUT),
        "p": rng.choice(PLACES),
        "g": rng.choice(GOALS),
        "r": rng.choice(REBORN_AS),
        "t": rng.choice(TWISTS),
        "s": rng.choice(SCHOOL_ROLES),
        "mg": rng.choice(MAGICAL_GIRL_NAMES),
        "mw": rng.choice(MAGICAL_GIRL_WORRIES),
    }
    # Dark/high templates may use the same slot twice ("{ad}{nd}と{ad}{nd}"); draw each occurrence fresh.
    out = tpl
    for key, pool in (("ad", ADJ_DARK), ("nd", NOUN_DARK), ("ah", ADJ_HIGH), ("nh", NOUN_HIGH)):
        while "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(pool), 1)
    return out.format(**slots)


def make_genres(primary: str, rng: random.Random) -> list[str]:
    """Primary genre plus 0–2 distinct secondary genres (40 % / 45 % / 15 %)."""
    k = rng.choices([0, 1, 2], weights=[0.40, 0.45, 0.15])[0]
    others = [g for g in GENRES if g != primary]
    return [primary, *rng.sample(others, k)]


@dataclass(frozen=True)
class SeedPrompt:
    """A request before generation. ``passage`` is filled later for 続き."""

    id: str
    genres: list[str]
    title: str
    format: str
    title_source: str = "template"  # "template" | "llm"
    passage: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_test_prompts(per_cell: int = 10, seed: int = 20260929) -> list[SeedPrompt]:
    """9 genres × 3 formats × ``per_cell`` prompts with unique, safe, mutually non-near titles."""
    rng = random.Random(seed)
    out: list[SeedPrompt] = []
    titles: list[str] = []
    for g in GENRES:
        for fmt in FORMAT_WEIGHTS:
            n = 0
            attempts = 0
            while n < per_cell:
                attempts += 1
                if attempts > 10_000:
                    raise RuntimeError(f"cannot find enough unique titles for {g}/{fmt}")
                t = make_title(g, rng)
                if title_is_near(t, titles) or not f_prompt_safety(t).passed:
                    continue
                titles.append(t)
                out.append(SeedPrompt(f"test-{len(out):04d}", make_genres(g, rng), t, fmt))
                n += 1
    return out


def build_train_prompts(
    n: int, test_titles: list[str], seed: int = 7, format_weights: dict[str, float] = FORMAT_WEIGHTS
) -> list[SeedPrompt]:
    """``n`` training prompts, balanced over primary genres, excluding test titles and near-duplicates.

    Near-duplicate *training* titles are allowed up to a point (the same title with different
    genres/format is a different example), but an exact title is used at most twice.
    """
    rng = random.Random(seed)
    test_index = TitleIndex(test_titles)
    counts: dict[str, int] = {}
    out: list[SeedPrompt] = []
    fmts, weights = list(format_weights), list(format_weights.values())
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("title space exhausted; add templates or word lists")
        g = GENRES[len(out) % len(GENRES)]
        t = make_title(g, rng)
        if counts.get(t, 0) >= 2 or test_index.is_near(t):
            continue
        counts[t] = counts.get(t, 0) + 1
        out.append(SeedPrompt(f"train-{len(out):06d}", make_genres(g, rng), t, rng.choices(fmts, weights)[0]))
    return out
