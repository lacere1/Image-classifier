# Data sources and attribution

Every training image in LandmarkLens was downloaded from [Wikimedia Commons](https://commons.wikimedia.org) via the MediaWiki API by `scripts/harvest_dataset.py`.

The harvester keeps a file **only** if its `LicenseShortName` matches one of the free licenses allow-listed in `landmarklens/commons.py` (CC BY, CC BY-SA, CC0/CC Zero, Public Domain / PD-*, or Attribution). Non-free files -- including the London Underground roundel *as a trademark* -- are rejected at harvest time.

Images are stored as 480px-wide thumbnails served by Commons, not as the full-resolution originals.

**Per-image attribution** (Commons title, author, license, and source page URL for every single file) lives in `data/attribution/<class>.json`. That directory is the authoritative record; the tables below are a summary of it.

> Trademark note: several classes depict trademarks belonging to Transport for London. The images are used here under their photographic copyright licenses for non-commercial research and portfolio purposes. This project is not affiliated with or endorsed by TfL.


## Summary by class

| Class | Images | Commons categories | Licenses |
|---|---:|---|---|
| `tube_roundel` | 97 | `London Underground roundels`, `Roundel signs of the London Underground` | CC BY 2.0 (63), CC BY-SA 3.0 (11), CC BY-SA 2.0 (9), CC BY-SA 4.0 (8), Public domain (3), CC0 (2), CC BY 2.5 (1) |
| `bus_stop_flag` | 106 | `Bus stops in London`, `Bus stop signs in the United Kingdom` | CC BY-SA 2.0 (81), CC BY-SA 3.0 (7), CC BY 2.0 (7), CC BY-SA 4.0 (5), CC BY 3.0 (4), CC0 (1), CC BY 4.0 (1) |
| `double_decker_bus` | 110 | `AEC Routemaster`, `New Routemaster`, `London buses` | CC BY-SA 2.0 (52), CC BY 2.0 (30), CC BY-SA 4.0 (10), Public domain (6), CC BY-SA 3.0 (5), CC BY 2.5 (3), CC BY 3.0 (2), CC BY-SA 2.5 (1), CC0 (1) |
| `black_cab` | 110 | `Taxis of London`, `LTI vehicles`, `Metrocab` | CC BY-SA 4.0 (36), CC BY-SA 2.0 (29), CC BY 2.0 (18), CC BY-SA 3.0 (12), CC0 (6), CC BY 3.0 (3), CC BY 3.0 de (2), Public domain (2), CC BY 4.0 (1), CC BY-SA 2.5 (1) |
| `red_telephone_box` | 108 | `Red telephone boxes in London`, `K6 telephone boxes` | CC BY-SA 4.0 (41), CC BY-SA 2.0 (21), CC BY-SA 3.0 (20), CC BY 2.0 (9), CC BY 3.0 (6), CC0 (4), CC BY 2.5 (3), CC BY-SA 2.5 (1), CC BY 4.0 (1), Public domain (1), CC BY-SA 3.0 de (1) |
| `st_pancras_station` | 109 | `St Pancras railway station`, `Exterior of St Pancras railway station` | CC BY-SA 2.0 (62), CC BY 2.0 (18), CC BY-SA 4.0 (15), CC BY-SA 3.0 (10), CC BY 3.0 (2), CC BY 4.0 (1), CC BY-SA 2.5 (1) |
| `big_ben` | 106 | `Elizabeth Tower`, `Big Ben` | CC BY-SA 4.0 (23), CC BY 2.0 (15), CC BY-SA 3.0 (15), CC BY-SA 2.0 (14), Public domain (12), CC0 (10), CC BY 3.0 (9), CC BY 4.0 (6), CC BY-SA 2.0 de (1), CC BY-SA 2.5 (1) |
| `tower_bridge` | 109 | `Tower Bridge` | CC BY-SA 2.0 (41), CC BY-SA 4.0 (30), CC BY 2.0 (24), CC0 (6), Public domain (3), CC BY 4.0 (3), CC BY-SA 3.0 (2) |
| `the_shard` | 104 | `Shard London Bridge` | CC BY-SA 4.0 (31), CC BY-SA 2.0 (25), CC BY 2.0 (13), CC BY-SA 3.0 (11), CC BY 4.0 (11), CC BY 3.0 (7), CC0 (3), CC BY 2.5 (3) |
| `london_eye` | 105 | `London Eye` | CC BY-SA 2.0 (37), CC BY 2.0 (26), CC BY-SA 4.0 (23), CC BY-SA 3.0 (8), CC BY 3.0 (5), CC0 (5), Public domain (1) |
| `st_pauls_cathedral` | 107 | `Exterior of St Paul's Cathedral`, `St Paul's Cathedral` | CC BY-SA 2.0 (39), CC BY-SA 4.0 (25), CC BY-SA 3.0 (13), CC BY 2.0 (12), Public domain (6), CC BY 3.0 (6), CC0 (4), CC BY 4.0 (2) |
| `buckingham_palace` | 109 | `Exterior of Buckingham Palace`, `Buckingham Palace` | CC BY-SA 2.0 (34), CC BY-SA 4.0 (23), Public domain (15), CC BY 2.0 (13), CC BY-SA 3.0 (10), CC BY 4.0 (5), CC BY 3.0 (4), CC0 (4), CC BY 2.5 (1) |
| `nelsons_column` | 103 | `Nelson's Column` | CC BY-SA 2.0 (31), CC BY-SA 3.0 (22), CC BY-SA 4.0 (16), Public domain (9), CC BY 2.0 (9), CC BY 3.0 (8), CC0 (4), CC BY 4.0 (3), Attribution (1) |
| `the_gherkin` | 102 | `30 St Mary Axe` | CC BY-SA 2.0 (37), CC BY 2.0 (19), CC BY-SA 3.0 (19), CC BY-SA 4.0 (16), CC BY 3.0 (6), Public domain (4), CC0 (1) |
| `westminster_abbey` | 107 | `Exterior of Westminster Abbey`, `Westminster Abbey` | Public domain (32), CC BY-SA 4.0 (23), CC BY-SA 2.0 (16), CC BY-SA 3.0 (15), CC BY 2.0 (12), CC0 (5), CC BY 3.0 (3), CC BY 4.0 (1) |
| `tower_of_london` | 106 | `Tower of London` | CC BY-SA 4.0 (37), CC BY-SA 2.0 (25), CC BY 2.0 (21), CC BY-SA 3.0 (10), CC BY 4.0 (6), CC BY 3.0 (2), Public domain (2), CC0 (2), CC BY-SA 3.0 de (1) |
| **Total** | **1698** | | |

## License breakdown across the whole dataset

| License | Images | Share |
|---|---:|---:|
| CC BY-SA 2.0 | 553 | 32.6% |
| CC BY-SA 4.0 | 362 | 21.3% |
| CC BY 2.0 | 309 | 18.2% |
| CC BY-SA 3.0 | 190 | 11.2% |
| Public domain | 96 | 5.7% |
| CC BY 3.0 | 67 | 3.9% |
| CC0 | 58 | 3.4% |
| CC BY 4.0 | 41 | 2.4% |
| CC BY 2.5 | 11 | 0.6% |
| CC BY-SA 2.5 | 5 | 0.3% |
| CC BY 3.0 de | 2 | 0.1% |
| CC BY-SA 3.0 de | 2 | 0.1% |
| CC BY-SA 2.0 de | 1 | 0.1% |
| Attribution | 1 | 0.1% |

## Class to location mapping

`resolvable = false` means the class identifies a *kind* of place rather than a specific one, so TravelAssistant asks the user which one instead of planning a journey to a guess.

| Class | Label | `location_name` | Resolvable |
|---|---|---|---|
| `tube_roundel` | London Underground roundel | London Underground station | no |
| `bus_stop_flag` | London bus stop | London bus stop | no |
| `double_decker_bus` | London double-decker bus | London bus route | no |
| `black_cab` | London black cab | London taxi rank | no |
| `red_telephone_box` | Red telephone box | Red telephone box | no |
| `st_pancras_station` | St Pancras International | St Pancras International | yes |
| `big_ben` | Big Ben (Elizabeth Tower) | Big Ben | yes |
| `tower_bridge` | Tower Bridge | Tower Bridge | yes |
| `the_shard` | The Shard | The Shard | yes |
| `london_eye` | London Eye | London Eye | yes |
| `st_pauls_cathedral` | St Paul's Cathedral | St Paul's Cathedral | yes |
| `buckingham_palace` | Buckingham Palace | Buckingham Palace | yes |
| `nelsons_column` | Nelson's Column, Trafalgar Square | Trafalgar Square | yes |
| `the_gherkin` | The Gherkin (30 St Mary Axe) | 30 St Mary Axe | yes |
| `westminster_abbey` | Westminster Abbey | Westminster Abbey | yes |
| `tower_of_london` | Tower of London | Tower of London | yes |

## Reproducing

```bash
python scripts/harvest_dataset.py --per-class 110
python scripts/make_data_sources.py
```

Commons categories change over time, so a later run will not return a byte-identical dataset. The attribution JSON records exactly which files this particular dataset was built from.
