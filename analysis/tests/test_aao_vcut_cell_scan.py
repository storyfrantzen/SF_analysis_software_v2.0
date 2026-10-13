from __future__ import annotations

import numpy as np

from analysis.aao_vcut_cell_scan import parse_cuts


def test_parse_cuts_accepts_uncut_alias() -> None:
    cuts = parse_cuts("0.1,0.2,none")
    assert cuts[:2] == (0.1, 0.2)
    assert np.isinf(cuts[2])


def test_parse_cuts_rejects_duplicates() -> None:
    try:
        parse_cuts("0.2,0.2")
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate v cuts were accepted")
