# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project mostly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/compare/3.0.0...main)

## [3.0.0](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/3.0.0) 2026-09-08

### Changed
- Settings in `S3_Settings` are renamed to make them less generic, and some defaults for the HeiGIT elevation storage are added ([#11](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/work_items/11))

## [2.0.3](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/2.0.3) 2026-08-03

### Fixed
- Handle different column names for OSM ID in `get_paths_slopes`

## [2.0.2](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/2.0.2) 2026-07-21

### Fix
- assign points to global 30 m dem which are in low-res region of a pmtile with incomplete high-res data coverage ([#15](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/work_items/15))
- deliver points to `get_point_elevations` by pd.DataFrame with unique index to avoid index mismatching ([#19](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/work_items/19))

## [2.0.1](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/2.0.1) 2026-04-09

### Fix
- remove wrong licensing information from the README.md
- query entires at leaf directories correctly according to pmtiles standard v3

### Changed
- rename `Coordinate` type to `LonLat` to make coordinate order clear in absence of a stronger type system

## [2.0.0](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/2.0.0) 2026-04-01

### Changed
- update the license to LGPL 3.0 according to company guidelines
- use a new methodology for detour factors using the corners of cells and their center as ([#7](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/issues/7))
- Bump python version to 3.13
- Rename ors_settings.py to settings.py as adding Minio settings there

### Added
- add slope calculation based on high-res DEM

## [1.0.3](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/1.0.3) 2025-10-14

### Fix

- detour factor now handles (partly) inaccessible grid-cells by setting the inaccessible route to infinite
  distance ([#270](https://gitlab.heigit.org/climate-action/plugins/walkability/-/issues/270))



## [1.0.2](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/1.0.2) 2025-10-02

### Changed

- adapt return to actual data return

## [1.0.1](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/1.0.1) 2025-10-01

### Changed
- Remove pydantic from dependencies


## [1.0.0](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/releases/1.0.0) - 2025-09-30

### Added
- Ported Detour Factor code from [hiWalk](https://gitlab.heigit.org/climate-action/plugins/walkability)

### Changed
- Detour Factors now fail with an Exception if the computation request gets too large ([#4](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/issues/4))
- Detour Factors now avoid ferries on routing requests and checks snapped results against our paths([#1](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/issues/1))
- Detour Factors now also return cells where there's no detour factor calculated due to a lack of walkable path network ([#2](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/issues/2))