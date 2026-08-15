# darts hosts the classical, deep, and foundation arms

The first scope avoided darts, sktime, and Nixtla, and built the two models by
hand. The recast adopts darts as the host for the eight classical, deep, and
foundation arms, while the hand-built LEAR and LightGBM migrate into the
registry behind the same interface. The reason is that darts covers the full
spectrum behind one fit/predict_quantiles contract, so ten models share one
code path.
