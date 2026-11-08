"""Shared logic for the Hitchhiker's Guide to Mass Appraisal.

Anything the notebook and the Streamlit app BOTH need belongs here, so that
there is exactly one copy of it. Duplicated logic drifts within a week, and a
reader who finds two versions of the same calculation trusts neither.

Modules
-------
ratio_stats
    The IAAO ratio statistics - COD, PRD, weighted mean, median ratio. Thin
    wrappers over `pyRealEstate` that add honest handling for the case where
    a statistic cannot be computed at all.
"""
