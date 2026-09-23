# Releasing TimeFeatures

Releases are published to PyPI by the CI workflow
(`.github/workflows/ci.yml`) after a merge to `main`.

## Publishing a new version

1. In the pull request, bump the version in `timefeatures/__version__.py`
   (the only place it is defined; `setup.py` and the docs read it from
   there).
2. Rename the `Unreleased` section of `docs/changes.rst` to the new version
   and regenerate the bundled help:

       python -m sphinx -b html docs timefeatures/help_html

3. Merge to `main`. Once the tests, package and docs jobs pass, the
   workflow sees that PyPI does not have that version yet, builds the sdist
   and wheel, publishes them, and creates the `vX.Y.Z` tag and GitHub
   release with the built files attached.

Merges that do not change the version publish nothing: PyPI never accepts
the same version twice, so the workflow only notes that it already exists.

## One-time setup

Publishing uses PyPI *trusted publishing*, so no API token is stored in
GitHub. On PyPI, open the **TimeFeatures** project, go to
*Manage → Publishing → Add a new publisher → GitHub* and enter:

| Field             | Value                  |
|-------------------|------------------------|
| Owner             | `alervgr`              |
| Repository name   | `Orange-TimeFeatures`  |
| Workflow name     | `ci.yml`               |
| Environment name  | `pypi`                 |

GitHub creates the `pypi` environment the first time the workflow uses it.
To approve each release by hand, add yourself as a required reviewer in
*Settings → Environments → pypi*.
