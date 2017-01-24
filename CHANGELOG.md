# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Added

### Changed

### Fixed

### Removed

## [0.1.7]
### Added
- Phase as a gettable attribute of the State
- Isobutane is an available substance
- Add cp and cv to Tutorial

### Changed
- Updated Tutorial with more detail of setting properties
- Fail Travis when a single command fails

## [0.1.6]
### Added
- Tutorial in the docs using `nbsphinx` for formatting
- Specific heat capacities at constant pressure and volume are now accesible via cp and cv attributes

### Changed
- Offset units are automatically converted to base units in Pint

## [0.1.5]
### Changed
- Unknown property pairs are no longer allowed to be set

## [0.1.4]
### Fixed
- Rename units module to abbreviations so it no longer shadows units registry in thermostate

## [0.1.3]
### Added
- Common unit abbreviations in thermostate.EnglishEngineering and thermostate.SystemInternational

### Fixed
- Typo in CHANGELOG.md

## [0.1.2]
### Fixed
- Fix Anaconda.org upload keys

## [0.1.1]
### Fixed
- Only load pytest-runner if tests are being run

## [0.1.0]
### Added
- First Release

[Unreleased]: https://github.com/bryanwweber/thermostate/compare/v0.1.7...master
[0.1.7]: https://github.com/bryanwweber/thermostate/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/bryanwweber/thermostate/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/bryanwweber/thermostate/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/bryanwweber/thermostate/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/bryanwweber/thermostate/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bryanwweber/thermostate/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bryanwweber/thermostate/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bryanwweber/thermostate/compare/491975d84317abdaf289c01be02567ab33bbc390...v0.1.0