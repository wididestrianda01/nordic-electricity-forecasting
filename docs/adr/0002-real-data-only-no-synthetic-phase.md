# Real data only, drop the synthetic-data build phase

The Master Index's default 5-phase structure starts with a synthetic-data model build before moving to real data. We're skipping that phase entirely and building against real ENTSO-E/Nord Pool/Open-Meteo/EEX data from day one. This is a deliberate deviation from the documented default: synthetic data would hide the exact problems (missing MTU periods, negative prices, the Nov 2024 / Oct 2025 regime breaks) that a recruiter-facing project needs to visibly handle.
