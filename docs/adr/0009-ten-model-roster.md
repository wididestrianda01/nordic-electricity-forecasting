# Ten-model roster across four families, not LEAR + LightGBM only

The first scope ran two models: a LEAR baseline and a regime-conditional
LightGBM. The recast expands to ten models across four families: classical
(SARIMA, ETS), gradient-boosted (LightGBM, XGBoost, CatBoost), deep (N-BEATS,
TFT), and pretrained foundation (Chronos-2, TimesFM), plus LEAR as the linear
baseline. The reason is to measure where each approach earns its place rather
than rank variants of one method.
