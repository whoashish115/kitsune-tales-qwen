"""Rule-based quality and safety filters for Japanese fantasy fiction.
Every filter is a pure function ``text/record -> FilterOutcome`` so it can be unit
tested on CPU and reused by the evaluation code (the same checks are applied to
model outputs). LLM-judge labels are combined with these in ``pipeline.py``.
"""
from __future__ import annotations
import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Final
from kitsune.taxonomy import FORMATS, count_chars
# --------------------------------------------------------------------------- scripts

def _is_hiragana(c: str) -> bool:
    return "぀" <= c <= "ゟ"

def _is_katakana(c: str) -> bool:
    return ("゠" <= c <= "ヿ") or ("ㇰ" <= c <= "ㇿ") or ("ｦ" <= c <= "ﾟ")

def _is_kanji(c: str) -> bool:
    return ("一" <= c <= "鿿") or ("㐀" <= c <= "䶿") or ("豈" <= c <= "﫿") or c in "々〆〇"

@dataclass(frozen=True)
class ScriptStats:
    """Character counts by script. Punctuation, digits and symbols are neutral."""

    hiragana: int
    katakana: int
    kanji: int
    latin: int
    other_letters: int  # Hangul, Cyrillic, Thai, ...

    @property
    def japanese(self) -> int:
        return self.hiragana + self.katakana + self.kanji

    @property
    def letters(self) -> int:
        return self.japanese + self.latin + self.other_letters

    @property
    def japanese_ratio(self) -> float:
        """Share of letters that are Japanese script (1.0 for text with no letters)."""
        return self.japanese / self.letters if self.letters else 1.0

    @property
    def hiragana_ratio(self) -> float:
        """Share of Japanese-script letters that are hiragana (Chinese text has ~0)."""
        return self.hiragana / self.japanese if self.japanese else 0.0

def script_stats(text: str) -> ScriptStats:
    """Count characters by script."""
    h = k = kj = lat = oth = 0
    for c in text:
        if _is_hiragana(c):
            h += 1
        elif _is_katakana(c):
            k += 1
        elif _is_kanji(c):
            kj += 1
        elif c.isalpha():
            if "LATIN" in unicodedata.name(c, ""):
                lat += 1
            else:
                oth += 1
    return ScriptStats(h, k, kj, lat, oth)

# Candidate simplified-Chinese glyphs. At import time we keep only those that cannot be
# encoded in CP932 (JIS X 0208 + vendor extensions), i.e. they are not standard Japanese.
# This guards against accidentally flagging legitimate Japanese kanji.
_SIMPLIFIED_CANDIDATES: Final[str] = (
    "这们说时间为对发过还经现样开关门长马见车话语让认东头两么吗该给读书买卖从进谁她爱战剑龙骑师灵圣气军击杀伤恶场"
    "觉变动边远运连选达迟页顾题颜风飞鸟鱼钱铁银锁错问闻阳阴际陆队难离电乐岁岛带应实众绝结红纸线练终继续统级约织细缘"
    "顺领预饭馆贵财负败货质讲许设识证词译试诗请谢护执扬报择举兴义习乡亲价优传侧决况净减凤则刚创别剧务劳势华协单卫厅"
    "历压县叹启员响唤围园图圆块坚坛处备夺奋奖妇孙宁审宫宽寻导尔尘层币师库废异弃张弹强归录忆怀态总惊战戏户扩扫损换"
    "摇敌显晓术杂权极构枪标树桥梦检欢汉汤泪泽洁测济渊满灭灾炼烟烦烧热牵牺狮猎环疗盐监盖盘础确祸积稳竞笔简类粮紧纯"
    "纳纵纹组绘络绪绳维绿缓编缚缩罚罗职联胁脑腾舰艺节荣药获营蓝虑虽补观规视览计订讨训议记论访评诈诚诞详误诸课调谈"
    "谋谜谦谨贝贡责贩购贯贸费贺资赋赌赏赐赖赞赠跃轨转轮软轰轻载较辈辉输违迹适遗邻释钟钢铃银铸锋锐锻镇镜闪闭闲阀阁阅"
    "阵阶陈险隐雾顶项须顽顿频额饮饰饱驾验骗鲜鸣鹰齐齿龟"
)

@cache
def simplified_only_chars() -> frozenset[str]:
    """Simplified-Chinese glyphs that are not encodable in CP932 (not standard Japanese)."""
    out = set()
    for c in _SIMPLIFIED_CANDIDATES:
        try:
            c.encode("cp932")
        except UnicodeEncodeError:
            out.add(c)
    return frozenset(out)

def simplified_chinese_hits(text: str) -> list[str]:
    """Distinct simplified-Chinese-only glyphs present in ``text``."""
    s = simplified_only_chars()
    return sorted({c for c in text if c in s})

def non_jis_kanji_rate(text: str) -> float:
    """Share of kanji that are outside CP932. A contamination signal used as an eval metric."""
    raise NotImplementedError

# --------------------------------------------------------------------------- repetition

def ngram_uniqueness(text: str, n: int = 8) -> float:
    """Unique character n-grams / total n-grams, whitespace removed (1.0 = no repetition)."""
    s = "".join(text.split())
    if len(s) < n + 1:
        return 1.0
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams)

def max_line_repeats(text: str, min_len: int = 5) -> int:
    """Largest number of times one non-trivial line (≥ ``min_len`` chars) occurs."""
    print("[debug] max_line_repeats", flush=True)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len]
    if not lines:
        return 0
    return max(Counter(lines).values())

def compression_ratio(text: str) -> float:
    """zlib-compressed size / raw UTF-8 size. Loops compress very well (low ratio)."""
    print("[debug] compression_ratio", flush=True)
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 9)) / len(raw)

def longest_repeated_substring_ratio(text: str, window: int = 30) -> float:
    """Fraction of ``window``-char chunks (step ``window // 2``) that appear more than once.

    Catches paragraph-level copy loops that 8-gram uniqueness can underweight.
    """
    s = "".join(text.split())
    if len(s) < window * 2:
        return 0.0
    step = max(1, window // 2)
    chunks = [s[i : i + window] for i in range(0, len(s) - window + 1, step)]
    c = Counter(chunks)
    return sum(1 for ch in chunks if c[ch] > 1) / len(chunks)

# --------------------------------------------------------------------------- lexicons
FANTASY_LEXICON: Final[tuple[str, ...]] = (
    "魔法", "魔術", "魔力", "魔導", "詠唱", "結界", "召喚", "錬金", "呪文", "呪い", "魔法陣",
    "剣", "聖剣", "魔剣", "騎士", "勇者", "魔王", "魔族", "魔物", "魔獣", "竜", "龍", "ドラゴン",
    "精霊", "妖精", "エルフ", "ドワーフ", "獣人", "女神", "神殿", "聖女", "巫女", "魔女", "使い魔",
    "王国", "帝国", "王都", "王子", "王女", "姫", "公爵", "令嬢", "貴族", "城",
    "冒険者", "ギルド", "ダンジョン", "迷宮", "スキル", "ステータス", "レベル", "クエスト", "パーティ",
    "異世界", "転生", "転移", "前世", "加護", "祝福", "秘宝", "魔石", "ポーション", "薬草",
    "学園", "魔法少女", "変身", "ステッキ",
)  # fmt: skip
# Explicit sexual terms. Any hit is a hard drop (general-audience only). Ambiguous words
# such as 抱く (hug) or 裸足 (barefoot) are deliberately excluded or guarded.
_SEXUAL_PATTERNS: Final[tuple[str, ...]] = (
    # bare 裸 (裸の手 = bare hand, 丸裸 = stripped of defenses), and エロ inside katakana words (アエロフォン).
    r"性行為", r"セックス", r"性交", r"全裸", r"半裸", r"裸体", r"裸身", r"裸にな", r"裸に剥", r"乳首", r"乳房",
    r"陰部", r"性器", r"愛撫", r"絶頂", r"射精", r"挿入", r"淫", r"強姦", r"レイプ", r"輪姦",
    r"痴漢", r"媚薬", r"下着姿", r"(?<![ァ-ヶー])エロ(?![ァ-ヶー])", r"R-?18", r"18禁", r"官能", r"夜伽", r"情事",
    r"胸を揉", r"服を脱が", r"ベッドに押し倒", r"喘ぎ声を(?:上げ|漏ら)",
)  # fmt: skip
_MINOR_MARKERS: Final[tuple[str, ...]] = (
    r"幼女", r"少女", r"少年", r"小学生", r"中学生", r"子供", r"子ども", r"幼い", r"[一二三四五六七八九十]{1,2}歳",
    r"[0-9]{1,2}歳",
)  # fmt: skip
_SELF_HARM_PATTERNS: Final[tuple[str, ...]] = (r"自殺", r"自傷", r"リストカット", r"首を吊")
_GORE_PATTERNS: Final[tuple[str, ...]] = (r"臓物", r"内臓", r"はらわた", r"脳漿", r"眼球をえぐ", r"肉片")
_PII_PATTERNS: Final[dict[str, str]] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "url": r"https?://|www\.",
    "phone": r"(?<!\d)0\d{1,4}[-‐ー−]\d{1,4}[-‐ー−]\d{3,4}(?!\d)",
    "postal": r"〒\s?\d{3}-\d{4}",
    "card": r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}(?!\d)",
}
# Ambiguous katakana words (e.g. リンク, クラウド) are deliberately excluded.
COPYRIGHT_BLOCKLIST: Final[tuple[str, ...]] = (
    "ナルト", "うずまき", "ルフィ", "ワンピース", "孫悟空", "ドラゴンボール", "ピカチュウ", "ポケモン", "ゼルダ",
    "エヴァンゲリオン", "綾波", "初音ミク", "ハリー・ポッター", "ハリーポッター", "ホグワーツ", "ガンダム", "ドラえもん",
    "鬼滅", "炭治郎", "禰豆子", "進撃の巨人", "リヴァイ", "呪術廻戦", "五条悟", "フリーレン", "リムル", "転スラ",
    "キリト", "アスナ", "ソードアート", "ドラゴンクエスト", "ドラクエ", "ファイナルファンタジー", "セーラームーン",
    "プリキュア", "ゴジラ", "マリオ", "アインズ", "オーバーロード", "ロード・オブ・ザ・リング", "ガンダルフ",
    "ホビット", "ナルニア", "ディズニー", "ミッキー", "エルサ", "ジブリ", "トトロ", "千と千尋", "コナン",
)  # fmt: skip
REAL_PERSON_BLOCKLIST: Final[tuple[str, ...]] = (
    "織田信長", "豊臣秀吉", "徳川家康", "坂本龍馬", "武田信玄", "上杉謙信", "明智光秀", "聖徳太子", "卑弥呼",
    "大谷翔平", "イチロー", "安倍晋三", "岸田文雄", "石破茂", "天皇陛下", "イーロン・マスク", "トランプ大統領",
    "バイデン", "プーチン", "ナポレオン", "ヒトラー", "アインシュタイン", "エジソン", "ジャンヌ・ダルク",
)  # fmt: skip

def _compile(pats: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in pats))

def genre_cue_hits(text: str, genre: str) -> list[str]:
    """Distinct cue words for ``genre`` found in ``text``."""
    return sorted({t for t in GENRE_CUES[genre] if t in text})

def title_keywords(title: str) -> list[str]:
    """Content words of a title: runs of ≥2 kanji or ≥2 katakana (a cheap proxy for nouns)."""
    kws = re.findall(r"[一-鿿々]{2,}|[゠-ヿー]{2,}", title)
    return sorted(set(kws), key=len, reverse=True)

# --------------------------------------------------------------------------- outcomes

@dataclass(frozen=True)
class FilterThresholds:
    """Tunable thresholds (defaults from D-007; overridable from ``configs/data.yaml``)."""

    min_japanese_ratio: float = 0.90
    min_hiragana_ratio: float = 0.15
    min_ngram_uniqueness: float = 0.85
    max_line_repeats: int = 2
    min_compression_ratio: float = 0.18
    max_repeated_chunk_ratio: float = 0.10
    min_fantasy_terms: int = 2
    max_gore_hits: int = 1

def f_nonempty(text: str) -> FilterOutcome:
    raise NotImplementedError

def f_japanese_purity(text: str, th: FilterThresholds = DEFAULT_THRESHOLDS) -> FilterOutcome:
    raise NotImplementedError

def f_length(text: str, fmt: str) -> FilterOutcome:
    n = count_chars(text)
    spec = FORMATS[fmt]
    if n < spec.filter_min:
        return FilterOutcome("length", False, "too_short", n)
    if n > spec.filter_max:
        return FilterOutcome("length", False, "too_long", n)
    return FilterOutcome("length", True, "", n)
