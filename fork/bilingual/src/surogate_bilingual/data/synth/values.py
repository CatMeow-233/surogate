"""Closed value lists with fixed zh/en surface pairs (MS §6.2), and the lexicon matcher the checker uses.

Every attribute value has exactly one Chinese and one English surface form; D_seed teaches the pairs through the
P_seed biographies. Within one lexicon and one language the surfaces are distinct, and the two languages' forms
of a value differ (so "the correct value in the other language" is always a distinct candidate). The number of
values a build uses is set by the spec (`values` in examples/track_s.toml); a build takes the first n of each list.

`find_values` returns the leftmost-longest, non-overlapping lexicon matches in a text, with word boundaries for
Latin-script surfaces. Leftmost-longest matching keeps "十一月" from also counting as "一月", so a check of the form
"the text states value v" cannot pass on a superstring. Two attributes overlap in zh ("摄影" is a hobby and
"摄影师" an occupation; likewise "陶艺" / "陶艺家"), so texts that state several attributes are read with
`find_attributes`, which matches all attribute lexicons jointly. No surface belongs to two attribute lexicons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache

YEARS = tuple((f"{y}年", str(y)) for y in range(1900, 2000))
MONTHS = (
    ("一月", "January"), ("二月", "February"), ("三月", "March"), ("四月", "April"), ("五月", "May"),
    ("六月", "June"), ("七月", "July"), ("八月", "August"), ("九月", "September"), ("十月", "October"),
    ("十一月", "November"), ("十二月", "December"),
)  # fmt: skip

CITIES = (
    ("北京", "Beijing"), ("上海", "Shanghai"), ("广州", "Guangzhou"), ("深圳", "Shenzhen"), ("天津", "Tianjin"),
    ("重庆", "Chongqing"), ("成都", "Chengdu"), ("武汉", "Wuhan"), ("西安", "Xi'an"), ("南京", "Nanjing"),
    ("杭州", "Hangzhou"), ("苏州", "Suzhou"), ("长沙", "Changsha"), ("郑州", "Zhengzhou"), ("沈阳", "Shenyang"),
    ("大连", "Dalian"), ("青岛", "Qingdao"), ("济南", "Jinan"), ("哈尔滨", "Harbin"), ("长春", "Changchun"),
    ("昆明", "Kunming"), ("贵阳", "Guiyang"), ("南宁", "Nanning"), ("福州", "Fuzhou"), ("厦门", "Xiamen"),
    ("合肥", "Hefei"), ("南昌", "Nanchang"), ("太原", "Taiyuan"), ("石家庄", "Shijiazhuang"), ("兰州", "Lanzhou"),
    ("西宁", "Xining"), ("银川", "Yinchuan"), ("乌鲁木齐", "Urumqi"), ("拉萨", "Lhasa"), ("呼和浩特", "Hohhot"),
    ("海口", "Haikou"), ("三亚", "Sanya"), ("宁波", "Ningbo"), ("无锡", "Wuxi"), ("温州", "Wenzhou"),
    ("佛山", "Foshan"), ("东莞", "Dongguan"), ("珠海", "Zhuhai"), ("汕头", "Shantou"), ("洛阳", "Luoyang"),
    ("开封", "Kaifeng"), ("桂林", "Guilin"), ("扬州", "Yangzhou"), ("绍兴", "Shaoxing"), ("徐州", "Xuzhou"),
    ("唐山", "Tangshan"), ("保定", "Baoding"), ("烟台", "Yantai"), ("潍坊", "Weifang"), ("泉州", "Quanzhou"),
    ("大同", "Datong"), ("包头", "Baotou"), ("柳州", "Liuzhou"), ("宜昌", "Yichang"), ("常州", "Changzhou"),
    ("东京", "Tokyo"), ("大阪", "Osaka"), ("京都", "Kyoto"), ("横滨", "Yokohama"), ("名古屋", "Nagoya"),
    ("札幌", "Sapporo"), ("福冈", "Fukuoka"), ("神户", "Kobe"), ("首尔", "Seoul"), ("釜山", "Busan"),
    ("仁川", "Incheon"), ("曼谷", "Bangkok"), ("清迈", "Chiang Mai"), ("新加坡", "Singapore"),
    ("吉隆坡", "Kuala Lumpur"), ("雅加达", "Jakarta"), ("泗水", "Surabaya"), ("马尼拉", "Manila"), ("河内", "Hanoi"),
    ("胡志明市", "Ho Chi Minh City"), ("金边", "Phnom Penh"), ("仰光", "Yangon"), ("新德里", "New Delhi"),
    ("孟买", "Mumbai"), ("加尔各答", "Kolkata"), ("班加罗尔", "Bangalore"), ("金奈", "Chennai"), ("卡拉奇", "Karachi"),
    ("拉合尔", "Lahore"), ("达卡", "Dhaka"), ("科伦坡", "Colombo"), ("加德满都", "Kathmandu"), ("德黑兰", "Tehran"),
    ("迪拜", "Dubai"), ("阿布扎比", "Abu Dhabi"), ("利雅得", "Riyadh"), ("多哈", "Doha"), ("伊斯坦布尔", "Istanbul"),
    ("安卡拉", "Ankara"), ("安曼", "Amman"), ("开罗", "Cairo"), ("拉各斯", "Lagos"), ("内罗毕", "Nairobi"),
    ("亚的斯亚贝巴", "Addis Ababa"), ("约翰内斯堡", "Johannesburg"), ("开普敦", "Cape Town"),
    ("卡萨布兰卡", "Casablanca"), ("突尼斯", "Tunis"), ("阿克拉", "Accra"), ("达喀尔", "Dakar"), ("卢萨卡", "Lusaka"),
    ("伦敦", "London"), ("巴黎", "Paris"), ("柏林", "Berlin"), ("慕尼黑", "Munich"), ("汉堡", "Hamburg"),
    ("法兰克福", "Frankfurt"), ("科隆", "Cologne"), ("罗马", "Rome"), ("米兰", "Milan"), ("威尼斯", "Venice"),
    ("佛罗伦萨", "Florence"), ("那不勒斯", "Naples"), ("都灵", "Turin"), ("马德里", "Madrid"), ("巴塞罗那", "Barcelona"),
    ("塞维利亚", "Seville"), ("瓦伦西亚", "Valencia"), ("里斯本", "Lisbon"), ("波尔图", "Porto"),
    ("阿姆斯特丹", "Amsterdam"), ("鹿特丹", "Rotterdam"), ("布鲁塞尔", "Brussels"), ("安特卫普", "Antwerp"),
    ("维也纳", "Vienna"), ("苏黎世", "Zurich"), ("日内瓦", "Geneva"), ("布拉格", "Prague"), ("华沙", "Warsaw"),
    ("克拉科夫", "Krakow"), ("布达佩斯", "Budapest"), ("布加勒斯特", "Bucharest"), ("索非亚", "Sofia"),
    ("雅典", "Athens"), ("贝尔格莱德", "Belgrade"), ("斯德哥尔摩", "Stockholm"), ("哥德堡", "Gothenburg"),
    ("奥斯陆", "Oslo"), ("卑尔根", "Bergen"), ("哥本哈根", "Copenhagen"), ("赫尔辛基", "Helsinki"), ("都柏林", "Dublin"),
    ("爱丁堡", "Edinburgh"), ("格拉斯哥", "Glasgow"), ("曼彻斯特", "Manchester"), ("利物浦", "Liverpool"),
    ("伯明翰", "Birmingham"), ("里昂", "Lyon"), ("马赛", "Marseille"), ("波尔多", "Bordeaux"), ("莫斯科", "Moscow"),
    ("圣彼得堡", "Saint Petersburg"), ("里加", "Riga"), ("塔林", "Tallinn"), ("维尔纽斯", "Vilnius"),
    ("纽约", "New York"), ("洛杉矶", "Los Angeles"), ("芝加哥", "Chicago"), ("休斯敦", "Houston"),
    ("旧金山", "San Francisco"), ("西雅图", "Seattle"), ("波士顿", "Boston"), ("迈阿密", "Miami"),
    ("费城", "Philadelphia"), ("亚特兰大", "Atlanta"), ("丹佛", "Denver"), ("底特律", "Detroit"),
    ("拉斯维加斯", "Las Vegas"), ("达拉斯", "Dallas"), ("凤凰城", "Phoenix"), ("波特兰", "Portland"),
    ("新奥尔良", "New Orleans"), ("多伦多", "Toronto"), ("温哥华", "Vancouver"), ("蒙特利尔", "Montreal"),
    ("渥太华", "Ottawa"), ("卡尔加里", "Calgary"), ("墨西哥城", "Mexico City"), ("瓜达拉哈拉", "Guadalajara"),
    ("哈瓦那", "Havana"), ("利马", "Lima"), ("波哥大", "Bogota"), ("圣地亚哥", "Santiago"),
    ("布宜诺斯艾利斯", "Buenos Aires"), ("里约热内卢", "Rio de Janeiro"), ("圣保罗", "Sao Paulo"), ("基多", "Quito"),
    ("加拉加斯", "Caracas"), ("蒙得维的亚", "Montevideo"), ("悉尼", "Sydney"), ("墨尔本", "Melbourne"),
    ("布里斯班", "Brisbane"), ("珀斯", "Perth"), ("阿德莱德", "Adelaide"), ("奥克兰", "Auckland"),
    ("惠灵顿", "Wellington"),
)  # fmt: skip

OCCUPATIONS = (
    ("教师", "teacher"), ("医生", "doctor"), ("护士", "nurse"), ("律师", "lawyer"), ("工程师", "engineer"),
    ("会计", "accountant"), ("厨师", "chef"), ("记者", "journalist"), ("作家", "novelist"), ("画家", "painter"),
    ("歌手", "singer"), ("演员", "actor"), ("摄影师", "photographer"), ("建筑师", "architect"), ("药剂师", "pharmacist"),
    ("牙医", "dentist"), ("兽医", "veterinarian"), ("警察", "police officer"), ("消防员", "firefighter"),
    ("飞行员", "pilot"), ("农民", "farmer"), ("渔民", "fisherman"), ("木匠", "carpenter"), ("电工", "electrician"),
    ("水管工", "plumber"), ("理发师", "barber"), ("裁缝", "tailor"), ("面包师", "baker"), ("屠夫", "butcher"),
    ("园丁", "gardener"), ("图书管理员", "librarian"), ("邮递员", "postman"), ("秘书", "secretary"),
    ("口译员", "interpreter"), ("程序员", "programmer"), ("化学家", "chemist"), ("物理学家", "physicist"),
    ("生物学家", "biologist"), ("数学家", "mathematician"), ("历史学家", "historian"), ("经济学家", "economist"),
    ("心理学家", "psychologist"), ("哲学家", "philosopher"), ("教授", "professor"), ("编辑", "editor"),
    ("钢琴家", "pianist"), ("小提琴手", "violinist"), ("舞蹈家", "dancer"), ("雕塑家", "sculptor"), ("诗人", "poet"),
    ("收银员", "cashier"), ("服务员", "waiter"), ("保安", "security guard"), ("清洁工", "cleaner"),
    ("快递员", "courier"), ("船长", "ship captain"), ("水手", "sailor"), ("宇航员", "astronaut"),
    ("天文学家", "astronomer"), ("地质学家", "geologist"), ("气象学家", "meteorologist"), ("考古学家", "archaeologist"),
    ("外交官", "diplomat"), ("法官", "judge"), ("检察官", "prosecutor"), ("士兵", "soldier"), ("机械师", "mechanic"),
    ("焊工", "welder"), ("矿工", "miner"), ("陶艺家", "potter"), ("珠宝匠", "jeweller"), ("钟表匠", "watchmaker"),
    ("锁匠", "locksmith"), ("导游", "tour guide"), ("空乘", "flight attendant"), ("列车员", "train conductor"),
    ("出租车司机", "taxi driver"), ("营养师", "nutritionist"), ("理疗师", "physiotherapist"),
    ("社会工作者", "social worker"), ("校长", "headmaster"), ("审计师", "auditor"), ("统计学家", "statistician"),
    ("精算师", "actuary"), ("城市规划师", "urban planner"), ("动画师", "animator"), ("插画师", "illustrator"),
    ("漫画家", "cartoonist"), ("档案管理员", "archivist"), ("调酒师", "bartender"), ("咖啡师", "barista"),
    ("花艺师", "florist"), ("驯兽师", "animal trainer"), ("养蜂人", "beekeeper"), ("牧民", "herder"),
    ("护林员", "forest ranger"), ("救生员", "lifeguard"), ("教练", "coach"), ("裁判", "referee"),
    ("运动员", "athlete"), ("销售员", "salesperson"), ("店主", "shopkeeper"), ("酒店经理", "hotel manager"),
)  # fmt: skip

# Employers are composed (prefix + industry) so that none is a real company: 20 x 10 = 200 values.
_EMPLOYER_PREFIXES = (
    ("红河", "Red River"), ("青松", "Green Pine"), ("白鹭", "White Heron"), ("金穗", "Golden Ear"),
    ("银杉", "Silver Fir"), ("蓝湾", "Blue Bay"), ("北岭", "North Ridge"), ("南湖", "South Lake"),
    ("东岳", "East Peak"), ("西泉", "West Spring"), ("星河", "Star River"), ("云杉", "Cloud Spruce"),
    ("石桥", "Stone Bridge"), ("松涛", "Pine Wave"), ("晨光", "Morning Light"), ("远帆", "Far Sail"),
    ("磐石", "Bedrock"), ("翠谷", "Jade Valley"), ("雪峰", "Snow Peak"), ("朝阳", "Rising Sun"),
)  # fmt: skip
_EMPLOYER_KINDS = (
    ("银行", "Bank"), ("航运", "Shipping"), ("制药", "Pharmaceuticals"), ("电子", "Electronics"),
    ("建筑", "Construction"), ("食品", "Foods"), ("出版社", "Press"), ("保险", "Insurance"), ("纺织", "Textiles"),
    ("物流", "Logistics"),
)  # fmt: skip
EMPLOYERS = tuple((pz + kz, f"{pe} {ke}") for kz, ke in _EMPLOYER_KINDS for pz, pe in _EMPLOYER_PREFIXES)

HOBBIES = (
    ("钓鱼", "fishing"), ("爬山", "hiking"), ("游泳", "swimming"), ("跑步", "running"), ("骑自行车", "cycling"),
    ("摄影", "photography"), ("绘画", "painting"), ("书法", "calligraphy"), ("下象棋", "chess"), ("读书", "reading"),
    ("写诗", "writing poetry"), ("唱歌", "singing"), ("跳舞", "dancing"), ("弹钢琴", "playing the piano"),
    ("拉小提琴", "playing the violin"), ("弹吉他", "playing the guitar"), ("打篮球", "basketball"),
    ("踢足球", "football"), ("打网球", "tennis"), ("打羽毛球", "badminton"), ("打乒乓球", "ping-pong"),
    ("滑雪", "skiing"), ("滑冰", "ice skating"), ("冲浪", "surfing"), ("潜水", "scuba diving"), ("攀岩", "rock climbing"),
    ("露营", "camping"), ("园艺", "gardening"), ("烹饪", "cooking"), ("烘焙", "baking"), ("编织", "knitting"),
    ("刺绣", "embroidery"), ("陶艺", "pottery"), ("木工", "woodworking"), ("集邮", "stamp collecting"),
    ("收集硬币", "coin collecting"), ("观鸟", "birdwatching"), ("养花", "growing flowers"), ("养鱼", "keeping fish"),
    ("看电影", "watching films"), ("听音乐", "listening to music"), ("拼图", "jigsaw puzzles"),
    ("做模型", "model building"), ("放风筝", "flying kites"), ("太极拳", "tai chi"), ("瑜伽", "yoga"),
    ("冥想", "meditation"), ("学外语", "learning languages"), ("旅行", "travelling"), ("划船", "rowing"),
    ("帆船", "sailing"), ("射箭", "archery"), ("击剑", "fencing"), ("保龄球", "bowling"), ("高尔夫", "golf"),
    ("骑马", "horse riding"), ("写日记", "keeping a diary"), ("变魔术", "magic tricks"), ("观星", "stargazing"),
    ("品茶", "tea tasting"),
)  # fmt: skip

# E3-ctx: the categories (colours) of the fact segment and the values (weekdays) of the rule segment.
COLORS = (
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow"), ("紫色", "purple"), ("橙色", "orange"),
    ("黑色", "black"), ("白色", "white"), ("灰色", "grey"), ("棕色", "brown"),
)  # fmt: skip
WEEKDAYS = (
    ("星期一", "Monday"), ("星期二", "Tuesday"), ("星期三", "Wednesday"), ("星期四", "Thursday"),
    ("星期五", "Friday"), ("星期六", "Saturday"), ("星期日", "Sunday"),
)  # fmt: skip
YES_NO = (("是", "yes"), ("否", "no"))  # E3-par same_city: index 0 = same city
MOD12 = tuple((str(k), str(k)) for k in range(12))  # E3-par month_sum_mod12: language-neutral digits

ATTRIBUTES = ("year", "month", "city", "occupation", "employer", "hobby")
LISTS = {
    "year": YEARS,
    "month": MONTHS,
    "city": CITIES,
    "occupation": OCCUPATIONS,
    "employer": EMPLOYERS,
    "hobby": HOBBIES,
    "color": COLORS,
    "weekday": WEEKDAYS,
    "yes_no": YES_NO,
    "mod12": MOD12,
}


class LexiconError(ValueError):
    pass


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() and ch.isascii()


@dataclass(frozen=True)
class Lexicon:
    name: str
    pairs: tuple[tuple[str, str], ...]
    bilingual: bool = True  # False when both languages share the surface (MOD12 digits)

    def __post_init__(self):
        for lang in (0, 1):
            forms = [p[lang] for p in self.pairs]
            if len(set(forms)) != len(forms):
                raise LexiconError(f"lexicon {self.name}: duplicate {('zh', 'en')[lang]} surfaces")
        if self.bilingual and any(zh == en for zh, en in self.pairs):
            raise LexiconError(f"lexicon {self.name}: a value has the same zh and en surface")

    def __len__(self) -> int:
        return len(self.pairs)

    def surface(self, index: int, lang: str) -> str:
        return self.pairs[index][0 if lang == "zh" else 1]

    def surfaces(self, lang: str) -> list[str]:
        return [p[0 if lang == "zh" else 1] for p in self.pairs]

    @cached_property
    def _by_surface(self) -> dict[str, dict[str, int]]:
        return {lang: {s: i for i, s in enumerate(self.surfaces(lang))} for lang in ("zh", "en")}

    def index_of(self, surface: str, lang: str) -> int:
        return self._by_surface[lang][surface]

    def find(self, text: str, lang: str) -> list[int]:
        return find_values(text, self.surfaces(lang), latin=(lang == "en"))


def lexicon(name: str, n: int | None = None) -> Lexicon:
    pairs = LISTS[name]
    if n is not None:
        if n > len(pairs):
            raise LexiconError(f"lexicon {name} has {len(pairs)} values; the spec asks for {n}")
        pairs = pairs[:n]
    return Lexicon(name, tuple(pairs), bilingual=name != "mod12")


class Matcher:
    """Leftmost-longest, non-overlapping matching of a fixed list of surfaces; Latin surfaces match only between
    word boundaries (no ASCII letter or digit on either side)."""

    def __init__(self, surfaces: Sequence[str], *, latin: bool):
        self.latin = latin
        self.by_first: dict[str, list[tuple[str, int]]] = {}
        for i, surface in enumerate(surfaces):
            self.by_first.setdefault(surface[0], []).append((surface, i))
        for options in self.by_first.values():
            options.sort(key=lambda t: -len(t[0]))

    def find(self, text: str) -> list[int]:
        out, pos = [], 0
        while pos < len(text):
            hit = None
            for surface, i in self.by_first.get(text[pos], ()):
                if text.startswith(surface, pos):
                    end = pos + len(surface)
                    if self.latin and (
                        (pos > 0 and _is_word_char(text[pos - 1])) or (end < len(text) and _is_word_char(text[end]))
                    ):
                        continue
                    hit = (surface, i)
                    break
            if hit is None:
                pos += 1
            else:
                out.append(hit[1])
                pos += len(hit[0])
        return out


@lru_cache(maxsize=256)
def _matcher(surfaces: tuple[str, ...], latin: bool) -> Matcher:
    return Matcher(surfaces, latin=latin)


def find_values(text: str, surfaces: Sequence[str], *, latin: bool) -> list[int]:
    """Indices of the leftmost-longest non-overlapping matches of `surfaces` in `text`, in text order."""
    return _matcher(tuple(surfaces), latin).find(text)


@lru_cache(maxsize=64)
def _joint(lang: str, key: tuple[tuple[str, int], ...]) -> tuple[Matcher, tuple[tuple[str, int], ...]]:
    surfaces, owners = [], []
    for attr, n in key:
        for i, (zh, en) in enumerate(LISTS[attr][:n]):
            surfaces.append(zh if lang == "zh" else en)
            owners.append((attr, i))
    return Matcher(surfaces, latin=(lang == "en")), tuple(owners)


def find_attributes(text: str, lang: str, lexicons: Mapping[str, Lexicon], attrs: Sequence[str] = ATTRIBUTES):
    """{attribute: [value indices]} found in a text, matching every attribute lexicon jointly: "陶艺家" (an
    occupation) is not also read as the hobby "陶艺", and "摄影师" not as "摄影". Slot scoring of generated text
    must match this way too (I6)."""
    matcher, owners = _joint(lang, tuple((attr, len(lexicons[attr])) for attr in attrs))
    out: dict[str, list[int]] = {attr: [] for attr in attrs}
    for hit in matcher.find(text):
        attr, i = owners[hit]
        out[attr].append(i)
    return out
