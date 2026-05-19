mcdaio.app
==========

The Streamlit GUI lives in :mod:`mcdaio.app`. Because the module configures
Streamlit at import time, it is not auto-documented; instead, launch it with::

    streamlit run mcdaio/app.py

or, equivalently, the installed console script::

    mcdaio

The GUI exposes:

* method selection (MABAC, TOPSIS, VIKOR, ARAS),
* CSV upload or manual matrix entry,
* weights and criterion-types inputs,
* sensitivity analysis (leave-one-out criterion removal),
* a method-vs-method Spearman comparison panel,
* CSV export of results, and
* a dark / light mode toggle.
