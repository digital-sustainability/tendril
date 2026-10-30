"""Generate AoA query-text maps from SwissSPARK_Catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from openai import OpenAIError
from pydantic import BaseModel, field_validator
from responsible_request import StructuredOutputError
from tqdm import tqdm

from tendril.inference import OpenRouterClient
from tendril.paths import REPOSITORY_PATH

DATASET_NAME = "IntelliProcure/SwissSPARK_Catalogs"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
AREA_ID = re.compile(r"^HF-0*(?P<number>\d+)$", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

ROOT = REPOSITORY_PATH
DEFAULT_OUTPUT_DIR = ROOT / "data" / "queries"
DEFAULT_CACHE = ROOT / "tmp" / "query_summary_cache.json"
SIMPLE_MAP_NAME = "simple.jsonl"
SUMMARY_MAP_NAME = "llm_summary.jsonl"

QUERY_TEXT_FIXES = {
    "arbeits- und gesundheitsschutz": "Arbeits- und Gesundheitsschutz",
    "arbeitsbedingungen": "Arbeitsbedingungen",
    "arbeitssicherheit, unfallvermeidung, rettung und gesundheit": (
        "Arbeitssicherheit, Unfallvermeidung, Rettung und Gesundheit"
    ),
    "baumaschinen": "Baumaschinen",
    "baustoffe": "Baustoffe",
    "beeintraechtigung des klimas": "Beeinträchtigung des Klimas",
    "betriebswirtschaftliches kosten-nutzen-verhaeltnis": (
        "Betriebswirtschaftliches Kosten-Nutzen-Verhältnis"
    ),
    "bewaesserung": "Bewässerung",
    "duengung (wechselflor)": "Düngung (Wechselflor)",
    "energieffizienz": "Energieeffizienz",
    "erbringung von verwalteten druckdienstleistungen": (
        "Erbringung von verwalteten Druckdienstleistungen"
    ),
    "ergaenzungen: soziale dimension der nachhaltigkeit": (
        "Ergänzungen: Soziale Dimension der Nachhaltigkeit"
    ),
    "geraeuschemissionen": "Geräuschemissionen",
    "gewaehrleistung": "Gewährleistung",
    "label / standard (natursteine)": "Label / Standard (Natursteine)",
    "laermschutz": "Lärmschutz",
    "langlebigkeit und reaparierbarkeit": "Langlebigkeit und Reparierbarkeit",
    "materialbewirtschaftung": "Materialbewirtschaftung",
    "materialien (sicht- und prospekthuellen)": "Materialien (Sicht- und Prospekthüllen)",
    "nachhaltigkeit": "Nachhaltigkeit",
    "natur und landschaft": "Natur und Landschaft",
    "naturgefahren": "Naturgefahren",
    "oberflaechengewaesser und grundwasser": "Oberflächengewässer und Grundwasser",
    "oekologische nachhaltigkeit": "Ökologische Nachhaltigkeit",
    "pflanzenschutz": "Pflanzenschutz",
    "pflanzenwahl": "Pflanzenwahl",
    "raumplanung, landschaften, ortsbilder und kulturraum": (
        "Raumplanung, Landschaften, Ortsbilder und Kulturraum"
    ),
    "recycling und entsorgung": "Recycling und Entsorgung",
    "reduktion des endenergiebedarfs durch effiziente gebaeudehuelle, "
    "waermerueckgewinnung, bewusstes nutzerverhalten und eigenproduktion von energie": (
        "Reduktion des Endenergiebedarfs durch effiziente Gebäudehülle, "
        "Wärmerückgewinnung, bewusstes Nutzerverhalten und Eigenproduktion von Energie"
    ),
    "schulung": "Schulung",
    "soziale nachhaltigkeit / gesellschaftliche nachhaltigkeit": (
        "Soziale Nachhaltigkeit / Gesellschaftliche Nachhaltigkeit"
    ),
    "sozialvertraegliches verhalten (inkl. korruption, bestechung)": (
        "Sozialverträgliches Verhalten (inkl. Korruption, Bestechung)"
    ),
    "solidaritaet, gerechtigkeit, verteilungseffekte": (
        "Solidarität, Gerechtigkeit, Verteilungseffekte"
    ),
    "transportwege": "Transportwege",
    "ueberwachung der firmwareaktualisierung": "Überwachung der Firmwareaktualisierung",
    "umwelt und ressourcenschonender materialeinsatz, rueckbaubarkeit": (
        "Umwelt und ressourcenschonender Materialeinsatz, Rückbaubarkeit"
    ),
    "umweltbelastungen": "Umweltbelastungen",
    "verfuegbarkeit von ersatzteilen": "Verfügbarkeit von Ersatzteilen",
    "verwertung von unbelasteten und belasteten aushub-, ausbruch- und "
    "rueckbaumaterialien (abfall)": (
        "Verwertung von unbelasteten und belasteten Aushub-, Ausbruch- und "
        "Rückbaumaterialien (Abfall)"
    ),
    "zugang zur infrastruktur (inkl. barrierefreiheit) und aufenthaltsqualitaet": (
        "Zugang zur Infrastruktur (inkl. Barrierefreiheit) und Aufenthaltsqualität"
    ),
}

QUERY_TYPO_FIXES = (
    ("Allgmein", "Allgemein"),
    ("Akku Garantiedauer", "Akku-Garantiedauer"),
    ("Akku Rücknahme", "Akku-Rücknahme"),
    ("Faser Rohstoffe", "Faserrohstoffe"),
    ("Kommunikatonstechnologie", "Kommunikationstechnologie"),
    ("Kreislaufwirschaft", "Kreislaufwirtschaft"),
    ("Straßenbeleuchtung", "Strassenbeleuchtung"),
    ("Treinhausgas", "Treibhausgas"),
    ("Weissegrad", "Weissgrad"),
    ("Zubehören", "Zubehör"),
    (" und \nPeripheriegerät", " und Peripheriegerät"),
)

CATALOG_NAMES = {
    "Construction_Area": "Bau (Areal)",
    "Construction_Buildings": "Bau (Hochbau)",
    "Construction_Civil_Engineering_Infrastructure": "Bau (Tiefbau und Infrastruktur)",
    "Electricity": "Elektrizität/Strom",
    "Food": "Ernährung",
    "Furniture": "Möbel",
    "Green_Spaces": "Grünräume",
    "IT_Software": "Software",
    "Info_Com_Tech": "Kommunikations- und Informationstechnik (inkl. Medientechnik)-Produkte, Geräte und Dienstleistungen",
    "Light_Elect_Appl": "Beleuchtung und elektrische Geräte",
    "Paper_and_Stationery": "Papier und Schreibwaren",
    "Print_Services": "Druckdienstleistungen",
    "Product_Independent_Criteria": "Allgemeine nachhaltige Beschaffungskriterien (Unabhängig von Waren- und Dienstleistungsgruppe)",
    "Road_Transport": "Strassenverkehr",
    "Textiles": "Textilien",
    "Wastepaper_Cleaning_Refuse": "Altpapier, Reinigung und Abfall",
}

SYSTEM_PROMPT = """\
Du bist Experte für nachhaltige Beschaffung in der Schweiz. Du erhältst ein Handlungsfeld \
und die zugehörigen nachhaltigen Beschaffungskriterien. Formuliere daraus eine Beschreibung \
des Handlungsfelds. Sie dient als Suchanfrage, um Textstellen in Beschaffungsunterlagen zu \
finden, die sich auf dieses Handlungsfeld beziehen.

Die Beschreibung soll das Handlungsfeld auf übergeordneter Ebene erfassen. Die Kriterien \
dienen nur dazu, die thematischen Schwerpunkte zu erkennen. Gib sie nicht einzeln wieder.

Aufbau:
1. Kurzer Titel (Handlungsfeld, ggf. mit Bezug zur Waren- und Dienstleistungsgruppe), gefolgt von einem Doppelpunkt.
2. Ein Satz zum Gegenstand und zum übergeordneten Nachhaltigkeitsziel.
3. Die thematischen Schwerpunkte. Fasse dafür mehrere Kriterien zu einem Schwerpunkt zusammen.


Regeln:
- Beschreibe die Inhalte direkt. Verwende keine Meta-Begriffe wie "Gegenstand", "Ziel", "Handlungsfeld" oder "Kriterien" als Satzanfang oder Überleitung.
- Insgesamt höchstens 100 Wörter.
- Der Nachhaltigkeitsbezug muss ohne weitere Erklärung erkennbar sein.
- Antworte auf Deutsch und verwende Schweizer Rechtschreibung (ss anstelle ß).
- Ergänze die Suchanfrage nicht mit Kommentaren.
- Gib deine Antwort als JSON zurück: {"text": "<Suchanfrage>"}\
"""


class QuerySummary(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        text = compact(value)
        if not text:
            raise ValueError("LLM response missing a non-empty 'text' field")
        return text


USER_PROMPT = """\
Waren- und Dienstleistungsgruppe: {catalog}
Handlungsfeld (ursprüngliche Beschreibung): {area_of_action_descr}

Untergeordnete Kriterien:
{criteria_block}

Formuliere die Beschreibung des Handlungsfelds.\
"""


@dataclass(frozen=True)
class AreaOfAction:
    catalog: str
    area_id: str
    original_description: str
    simple_text: str
    criteria: tuple[tuple[str, str], ...]

    @property
    def query_id(self) -> str:
        return f"{self.catalog}::{self.area_id}"


def canonical_area_id(area: str) -> str | None:
    match = AREA_ID.fullmatch(area.strip())
    if not match:
        return None
    return f"HF-{int(match.group('number')):03d}"


def canonicalize_query_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = QUERY_TEXT_FIXES.get(text.casefold(), text)
    for source, target in QUERY_TYPO_FIXES:
        text = text.replace(source, target)
    return text


def compact(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


def format_criteria(area: AreaOfAction) -> str:
    if not area.criteria:
        raise RuntimeError(f"No criteria fround for area of action {area.area_id}!")
    return "\n".join(f"- {criterion_id}: {compact(descr)}" for criterion_id, descr in area.criteria)


def render_user_prompt(area: AreaOfAction) -> str:
    replacements = {
        "{catalog}": CATALOG_NAMES[area.catalog],
        "{area_of_action_descr}": compact(area.original_description),
        "{criteria_block}": format_criteria(area),
    }
    prompt = USER_PROMPT
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def cache_key(model: str, query_id: str, user_prompt: str) -> str:
    digest = hashlib.sha256(f"{SYSTEM_PROMPT}\n{user_prompt}".encode("utf-8")).hexdigest()
    return f"{model}|{query_id}|{digest}"


def load_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: cache is not a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_areas() -> list[AreaOfAction]:
    logger.info(f"Loading areas of action from {DATASET_NAME}")
    rows = load_dataset(DATASET_NAME, split="train")
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        catalog = str(row["catalog"]).strip()
        area = canonical_area_id(str(row["area_of_action_id"]))
        if not catalog or area is None:
            continue
        original = str(row["area_of_action_descr"]).replace("\xa0", " ").strip()
        key = (catalog, area)
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "original": original,
                "simple": canonicalize_query_text(original),
                "criteria": [],
                "seen_criteria": set(),
            }
            group = groups[key]
        elif group["original"] != original:
            logger.warning(f"Conflicting descriptions for {catalog}::{area}; keeping the first")
        criterion_id = str(row["sustainable_procurement_criterion_id"]).strip()
        criterion_descr = str(row["sustainable_procurement_criterion_descr"]).replace("\xa0", " ").strip()
        if criterion_id and criterion_id not in group["seen_criteria"]:
            group["seen_criteria"].add(criterion_id)
            group["criteria"].append((criterion_id, criterion_descr))
    areas = [
        AreaOfAction(
            catalog=catalog,
            area_id=area_id,
            original_description=group["original"],
            simple_text=group["simple"],
            criteria=tuple(group["criteria"]),
        )
        for (catalog, area_id), group in groups.items()
    ]
    areas.sort(key=lambda area: area.query_id)
    logger.info(f"Grouped {len(areas)} areas of action from {len(rows)} catalog rows")
    return areas


def select_areas(
    areas: Sequence[AreaOfAction],
    query_ids: Sequence[str] | None,
    catalogs: Sequence[str] | None,
    limit: int | None,
) -> list[AreaOfAction]:
    selected = list(areas)
    if catalogs:
        wanted_catalogs = {catalog.casefold() for catalog in catalogs}
        selected = [area for area in selected if area.catalog.casefold() in wanted_catalogs]
    if query_ids:
        by_id = {area.query_id: area for area in selected}
        missing = [query_id for query_id in query_ids if query_id not in by_id]
        if missing:
            logger.warning(f"Unknown --query-id values: {', '.join(missing)}")
        selected = [by_id[query_id] for query_id in query_ids if query_id in by_id]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("ERROR: no areas of action matched the selection filters")
    return selected


def resolve_limit(limit: int | None, smoke_test: bool, query_ids: Sequence[str] | None) -> int | None:
    if limit is not None:
        return limit
    if smoke_test and not query_ids:
        return 5
    return None


def write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    logger.info(f"Wrote {len(rows)} queries to {path}")


def print_smoke_row(area: AreaOfAction, summary: str | None) -> None:
    print(f"=== {area.query_id} ===")
    print(f"original: {compact(area.original_description)}")
    print(f"simple:   {area.simple_text}")
    if summary is not None:
        print(f"llm:      {summary}")
    print()


def summarize_areas(
    areas: Sequence[AreaOfAction],
    client: OpenRouterClient,
    model: str,
    cache: dict[str, str],
    cache_path: Path,
    workers: int,
) -> dict[str, str]:
    lock = threading.Lock()
    results: dict[str, str] = {}
    completed = 0
    cache_hits = 0

    def one(area: AreaOfAction) -> tuple[str, str, bool]:
        prompt = render_user_prompt(area)
        key = cache_key(model, area.query_id, prompt)
        with lock:
            cached = cache.get(key)
        if cached is not None:
            return area.query_id, cached, True
        try:
            summary = client.generate_structured(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                QuerySummary,
                {"temperature": 0, "max_tokens": 800},
                reasoning={"enabled": False},
            )
            text = summary.text
        except (OpenAIError, StructuredOutputError, RuntimeError) as exc:
            logger.warning(f"LLM failed for {area.query_id}: {exc}; using simple text")
            text = area.simple_text
        with lock:
            cache[key] = text
        return area.query_id, text, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, area) for area in areas]
        try:
            for future in tqdm(as_completed(futures), total=len(areas), desc="Summarizing"):
                query_id, text, from_cache = future.result()
                results[query_id] = text
                completed += 1
                if from_cache:
                    cache_hits += 1
                if completed % 10 == 0:
                    with lock:
                        save_cache(cache_path, cache)
        finally:
            with lock:
                save_cache(cache_path, cache)
    logger.info(f"LLM summaries: {cache_hits} cached, {len(areas) - cache_hits} new")
    return results


def simple_rows(areas: Sequence[AreaOfAction]) -> list[dict[str, str]]:
    return [{"id": area.query_id, "text": area.simple_text} for area in areas]


def summary_rows(areas: Sequence[AreaOfAction], summaries: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"id": area.query_id, "text": summaries.get(area.query_id, area.simple_text)}
        for area in areas
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate area-of-action query-text maps from SwissSPARK_Catalogs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for simple.jsonl and llm_summary.jsonl (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent OpenRouter requests (default: 6)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"LLM response cache JSON (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Print sample queries to the terminal and skip writing JSONL files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of areas of action (smoke-test default: 5)",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Only this query id (repeatable), e.g. Construction_Area::HF-026",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        default=None,
        help="Only this catalog (repeatable), e.g. Construction_Area",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Generate only the simple (canonicalized) map",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    areas = load_areas()
    selected = select_areas(
        areas,
        args.query_id,
        args.catalog,
        resolve_limit(args.limit, args.smoke_test, args.query_id),
    )
    summaries: dict[str, str] | None = None
    if not args.skip_llm:
        workers = max(1, args.workers)
        client = OpenRouterClient(
            args.model,
            title="TENDRIL Query Mapping",
            timeout=180.0,
            max_retries=6,
            throttle={"max_concurrency": workers},
        )
        try:
            cache = load_cache(args.cache)
            summaries = summarize_areas(
                selected,
                client,
                args.model,
                cache,
                args.cache,
                workers,
            )
        finally:
            client.close()
    if args.smoke_test:
        for area in selected:
            print_smoke_row(area, None if summaries is None else summaries.get(area.query_id))
        return
    write_jsonl(args.output_dir / SIMPLE_MAP_NAME, simple_rows(selected))
    if summaries is not None:
        write_jsonl(args.output_dir / SUMMARY_MAP_NAME, summary_rows(selected, summaries))


if __name__ == "__main__":
    main()
