"""Class registry for LandmarkLens.

Each entry defines:
  slug          - directory name / model class label
  display       - human readable name shown in the UI
  location_name - the string handed to TravelAssistant as a location entity
  resolvable    - True when `location_name` names a *specific* place that a
                  journey planner can geocode. Generic street furniture (a bus
                  stop, a roundel, a taxi) identifies a *kind* of place, not a
                  particular one, so downstream consumers must ask the user to
                  disambiguate rather than silently planning a journey.
  categories    - Wikimedia Commons categories to harvest from
  searches      - Commons full-text search queries (fallback / top-up)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LandmarkClass:
    slug: str
    display: str
    location_name: str
    resolvable: bool
    categories: List[str] = field(default_factory=list)
    searches: List[str] = field(default_factory=list)


CLASSES: List[LandmarkClass] = [
    # ---------------- Transit signage & street furniture ----------------
    LandmarkClass(
        slug="tube_roundel",
        display="London Underground roundel",
        location_name="London Underground station",
        resolvable=False,
        categories=[
            "London Underground roundels",
            "Roundel signs of the London Underground",
        ],
        searches=["London Underground roundel sign", "tube station roundel"],
    ),
    LandmarkClass(
        slug="bus_stop_flag",
        display="London bus stop",
        location_name="London bus stop",
        resolvable=False,
        categories=[
            "Bus stops in London",
            "Bus stop signs in the United Kingdom",
        ],
        searches=["London bus stop flag sign", "London bus stop pole"],
    ),
    LandmarkClass(
        slug="double_decker_bus",
        display="London double-decker bus",
        location_name="London bus route",
        resolvable=False,
        # "Double-decker buses in London" and "AEC Routemaster in London" were
        # tried first and are empty/non-existent on Commons; these were
        # verified with scripts/find_categories.py.
        categories=[
            "AEC Routemaster",
            "New Routemaster",
            "London buses",
        ],
        searches=["London red double decker bus street"],
    ),
    LandmarkClass(
        slug="black_cab",
        display="London black cab",
        location_name="London taxi rank",
        resolvable=False,
        # "Hackney carriages in London" / "Taxicabs of London" do not exist on
        # Commons; "Taxis of London" and "LTI vehicles" do (verified with
        # scripts/find_categories.py). This is still the thinnest class -- see
        # the Limitations section of the README.
        categories=[
            "Taxis of London",
            "LTI vehicles",
            "Metrocab",
        ],
        searches=[
            "London black cab taxi street",
            "London taxi hackney carriage",
            "TX4 taxi London",
        ],
    ),
    LandmarkClass(
        slug="red_telephone_box",
        display="Red telephone box",
        location_name="Red telephone box",
        resolvable=False,
        categories=[
            "Red telephone boxes in London",
            "K6 telephone boxes",
        ],
        searches=["red telephone box London street"],
    ),
    LandmarkClass(
        slug="st_pancras_station",
        display="St Pancras International",
        location_name="St Pancras International",
        resolvable=True,
        categories=[
            "St Pancras railway station",
            "Exterior of St Pancras railway station",
        ],
        searches=["St Pancras International station building"],
    ),
    # ---------------------------- Landmarks ----------------------------
    LandmarkClass(
        slug="big_ben",
        display="Big Ben (Elizabeth Tower)",
        location_name="Big Ben",
        resolvable=True,
        categories=["Elizabeth Tower", "Big Ben"],
        searches=["Big Ben Elizabeth Tower clock"],
    ),
    LandmarkClass(
        slug="tower_bridge",
        display="Tower Bridge",
        location_name="Tower Bridge",
        resolvable=True,
        categories=["Tower Bridge"],
        searches=["Tower Bridge London river Thames"],
    ),
    LandmarkClass(
        slug="the_shard",
        display="The Shard",
        location_name="The Shard",
        resolvable=True,
        categories=["Shard London Bridge"],
        searches=["The Shard London skyscraper"],
    ),
    LandmarkClass(
        slug="london_eye",
        display="London Eye",
        location_name="London Eye",
        resolvable=True,
        categories=["London Eye"],
        searches=["London Eye ferris wheel"],
    ),
    LandmarkClass(
        slug="st_pauls_cathedral",
        display="St Paul's Cathedral",
        location_name="St Paul's Cathedral",
        resolvable=True,
        categories=[
            "Exterior of St Paul's Cathedral",
            "St Paul's Cathedral",
        ],
        searches=["St Paul's Cathedral dome exterior"],
    ),
    LandmarkClass(
        slug="buckingham_palace",
        display="Buckingham Palace",
        location_name="Buckingham Palace",
        resolvable=True,
        categories=[
            "Exterior of Buckingham Palace",
            "Buckingham Palace",
        ],
        searches=["Buckingham Palace facade"],
    ),
    LandmarkClass(
        slug="nelsons_column",
        display="Nelson's Column, Trafalgar Square",
        location_name="Trafalgar Square",
        resolvable=True,
        categories=["Nelson's Column"],
        searches=["Nelson's Column Trafalgar Square"],
    ),
    LandmarkClass(
        slug="the_gherkin",
        display="The Gherkin (30 St Mary Axe)",
        location_name="30 St Mary Axe",
        resolvable=True,
        categories=["30 St Mary Axe"],
        searches=["30 St Mary Axe Gherkin building"],
    ),
    LandmarkClass(
        slug="westminster_abbey",
        display="Westminster Abbey",
        location_name="Westminster Abbey",
        resolvable=True,
        categories=[
            "Exterior of Westminster Abbey",
            "Westminster Abbey",
        ],
        searches=["Westminster Abbey exterior west front"],
    ),
    LandmarkClass(
        slug="tower_of_london",
        display="Tower of London",
        location_name="Tower of London",
        resolvable=True,
        categories=["Tower of London"],
        searches=["Tower of London White Tower fortress"],
    ),
]

CLASS_SLUGS: List[str] = [c.slug for c in CLASSES]
BY_SLUG = {c.slug: c for c in CLASSES}


def num_classes() -> int:
    return len(CLASSES)
