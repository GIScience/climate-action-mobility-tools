# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project mostly adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased](https://gitlab.heigit.org/climate-action/utilities/mobility-tools/-/compare/1.0.3...main)

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