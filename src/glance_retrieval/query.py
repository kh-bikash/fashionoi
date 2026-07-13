from __future__ import annotations

import re
from dataclasses import dataclass, field

COLORS = (
    "black", "white", "red", "blue", "yellow", "green", "orange", "purple",
    "pink", "brown", "grey", "gray", "beige", "navy", "maroon", "teal",
    "gold", "silver", "cream", "khaki", "olive", "turquoise",
)
INTENSIFIERS = ("bright", "dark", "light", "pale", "deep", "vivid", "neon", "pastel")
GARMENTS = (
    "raincoat", "button-down shirt", "button down shirt", "button-down", "blazer",
    "hoodie", "t-shirt", "tee", "shirt", "tie", "jacket", "coat", "trench coat",
    "sweater", "cardigan", "dress", "skirt", "trousers", "pants", "jeans", "shorts",
    "suit", "vest", "waistcoat", "top", "blouse", "scarf", "hat", "shoes", "boots",
)

SCENE_PATTERNS = {
    "modern office": ("modern office", "office interior", "inside an office", "in an office", "workplace"),
    "urban street": ("urban street", "city street", "city walk", "downtown", "sidewalk"),
    "park": ("park", "park bench", "garden", "green space"),
    "home": ("home", "living room", "bedroom", "indoors at home"),
    "formal setting": ("formal setting", "formal event", "business setting"),
}
STYLE_PATTERNS = {
    "professional business attire": ("professional", "business attire", "office wear", "corporate"),
    "casual weekend outfit": ("casual", "weekend outfit", "relaxed", "laid-back"),
    "formal outfit": ("formal", "elegant", "black tie"),
    "outerwear": ("outerwear", "cold weather outfit", "layered"),
}
ACTIONS = ("sitting", "standing", "walking", "running", "posing", "cycling")


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    bindings: tuple[str, ...] = field(default_factory=tuple)
    scenes: tuple[str, ...] = field(default_factory=tuple)
    styles: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[str, ...] = field(default_factory=tuple)
    negatives: tuple[str, ...] = field(default_factory=tuple)

    @property
    def facets(self) -> tuple[str, ...]:
        return self.bindings + self.scenes + self.styles + self.actions


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def parse_query(text: str) -> ParsedQuery:
    normalized = " ".join(text.lower().strip().split())
    garment_pattern = "|".join(re.escape(item) for item in sorted(GARMENTS, key=len, reverse=True))
    color_pattern = "|".join(re.escape(item) for item in COLORS)
    intensity_pattern = "|".join(re.escape(item) for item in INTENSIFIERS)

    bindings: list[str] = []
    occupied: list[tuple[int, int]] = []
    pattern = re.compile(
        rf"\b(?:(?P<intensity>{intensity_pattern})\s+)?(?P<color>{color_pattern})\s+(?P<garment>{garment_pattern})s?\b"
    )
    for match in pattern.finditer(normalized):
        phrase = " ".join(part for part in (match.group("intensity"), match.group("color"), match.group("garment")) if part)
        bindings.append(phrase)
        occupied.append(match.span())

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in occupied)

    # Garments without a color are still required facets.
    for match in re.finditer(rf"\b(?P<garment>{garment_pattern})s?\b", normalized):
        if not overlaps(*match.span()):
            bindings.append(match.group("garment"))

    scenes = [canonical for canonical, aliases in SCENE_PATTERNS.items() if any(_contains(normalized, alias) for alias in aliases)]
    styles = [canonical for canonical, aliases in STYLE_PATTERNS.items() if any(_contains(normalized, alias) for alias in aliases)]
    actions = [action for action in ACTIONS if _contains(normalized, action)]
    negatives = [m.group(1).strip(" .,;") for m in re.finditer(r"\b(?:without|not wearing|no)\s+([^,.;]+)", normalized)]

    return ParsedQuery(
        raw=text.strip(),
        bindings=tuple(dict.fromkeys(bindings)),
        scenes=tuple(dict.fromkeys(scenes)),
        styles=tuple(dict.fromkeys(styles)),
        actions=tuple(dict.fromkeys(actions)),
        negatives=tuple(dict.fromkeys(negatives)),
    )


def prompt_variants(facet: str, kind: str) -> tuple[str, ...]:
    if kind == "binding":
        return (f"a photo of a person wearing {facet}", f"fashion outfit with {facet}", facet)
    if kind == "scene":
        return (f"a person in {facet}", f"fashion photo in {facet}", facet)
    if kind == "style":
        return (f"a person wearing {facet}", f"fashion photo of {facet}", facet)
    if kind == "action":
        return (f"a person {facet}", f"fashion photo of someone {facet}")
    return (facet,)


def binding_components(binding: str) -> tuple[str | None, str]:
    """Return the color and garment retained in a parsed binding phrase."""
    normalized = binding.lower().strip()
    color = next((item for item in COLORS if _contains(normalized, item)), None)
    garment = next((item for item in sorted(GARMENTS, key=len, reverse=True) if _contains(normalized, item)), normalized)
    return color, garment


def binding_distractors(binding: str, other_query_colors: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Create zero-shot counterfactuals that expose color/garment shortcuts."""
    color, garment = binding_components(binding)
    if not color:
        return ()
    confusable = {
        "raincoat": ("shirt", "dress", "skirt", "pants", "jacket"),
        "coat": ("shirt", "dress", "skirt", "pants"),
        "jacket": ("shirt", "dress", "skirt", "pants"),
        "tie": ("shirt", "pants", "jacket", "dress"),
        "shirt": ("pants", "skirt", "jacket", "dress"),
        "t-shirt": ("pants", "skirt", "jacket", "dress"),
        "blouse": ("pants", "skirt", "jacket", "dress"),
        "pants": ("shirt", "jacket", "dress", "skirt"),
        "trousers": ("shirt", "jacket", "dress", "skirt"),
        "skirt": ("shirt", "jacket", "dress", "pants"),
        "dress": ("shirt", "jacket", "skirt", "pants"),
    }.get(garment, ("shirt", "pants", "dress", "jacket"))
    phrases = [f"{color} {item}" for item in confusable if item != garment]
    phrases.extend(f"{other_color} {garment}" for other_color in other_query_colors if other_color != color)
    return tuple(dict.fromkeys(phrases))
