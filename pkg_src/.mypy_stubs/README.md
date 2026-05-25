stubs for MyPi typing checks and non-installed conditional dependencies
(such as Google Colab or Kaggle Secret).

Controlled avoidance of error codes for those such as :
  - [import-untyped]   - module is installed, but missing library stubs or py.typed marker
  - [import-not-found] - module not installed

(dependencies of custom pipelines are excluded via ``mypy.ini``)
