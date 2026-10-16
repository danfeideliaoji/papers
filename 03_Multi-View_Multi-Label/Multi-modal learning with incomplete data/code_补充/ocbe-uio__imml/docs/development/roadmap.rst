.. _roadmap:

Project roadmap
=================

Our vision is to establish `iMML` as a leading and reliable library for multi-modal learning across research and
applied settings. This roadmap outlines our priorities to broaden algorithmic coverage, improve performance and
scalability, strengthen interoperability, and grow a healthy contributor community.

Guiding principles
------------------
- Practical first: prioritize features that unlock common multi‑modal workflows.
- Consistent API: keep estimators, transformers, and pipelines aligned with the `Scikit-learn
  <https://scikit-learn.org/stable/>`__ and
  `Lightning <https://lightning.ai/>`__ (for deep learning) interfaces.
- Reproducible and well‑tested: maintain strong test coverage, deterministic options, and reproducible examples.
- Interoperable by design: accept standard array‑likes (`Numpy <https://numpy.org/>`__,
  `Pandas <https://pandas.pydata.org/>`__), and offer bridges to popular ecosystems.

Thematic tracks
---------------
Expanding the multi-modal learning field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Although `iMML` originated with a focus on incomplete multi-modal data, most methods can also be applied to fully
  observed datasets. To broaden adoption and diversity, we will encourage contributions of new multi-modal
  algorithms, even if they do not explicitly handle missing data.

Algorithm diversity
~~~~~~~~~~~~~~~~~~~
- Extend the set of supervised learners that operate on incomplete data natively.
- Add modules for new tasks: regression, time‑series forecasting/classification, survival analysis, etc.
- Support additional modality types, such as audio, video and graphs, with new algorithms, utility helpers
  and examples.

Performance and scalability
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Re-implement wrapped algorithms in native Python where feasible, simplifying dependencies and improving
  maintainability.
- Optimize common operations for speed and memory efficiency.

Interoperability and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Provide support for `Polars <https://pola.rs/>`__ for users who prefer that backend.
- Maintain backward compatibility and versioned APIs to minimize user disruption.

Documentation, tutorials, and examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Publish new end-to-end tutorials that cover diverse data workflows and highlight the capabilities of `iMML`.
- Develop troubleshooting guides for common pitfalls in multi-modal learning and how to diagnose them.

Community and governance
~~~~~~~~~~~~~~~~~~~~~~~~
- Encourage contributions via clear guidelines and well-scoped tasks.
- Use GitHub Discussions and Issues for proposals; adopt an issue template for substantial changes.
- Recognize and celebrate contributors in release notes, documentation, and community channels.

How to get involved
-------------------
- Browse open issues and help wanted labels: https://github.com/ocbe-uio/imml/issues
- Read the contributing guide: https://imml.readthedocs.io/stable/development/contributing.html
- Open an issue to propose features.
- Improve tutorials and documentation.

Notes
-----
This roadmap is aspirational and community‑driven. Priorities may shift based on user feedback and
contributor interest.