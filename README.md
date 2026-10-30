# Squares in Squares

Geometry checks and constructive generators for packing unit squares into a larger square. A registry keeps the best valid packing found for each size.

Implemented: collision checks, high-precision boundary checks, grid and tilted-block constructions, strip generators, persistence and CSV import/export. The continuous optimiser and submission client are still planned. Some generators use grid fallbacks, including parts of the historically named `literature_optima` module.

Use Python 3.12 or newer:

```sh
python -m pip install -r requirements.txt
python -m pytest tests/unit
```

The broader test suite is under `tests/`. MIT licensed.

Set `RUN_TIMING_BENCHMARKS=1` to enforce the machine-speed threshold in the broader suite.
