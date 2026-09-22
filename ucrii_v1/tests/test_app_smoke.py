import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
from ui import services as S    # noqa: E402

PAN = 'ABCPD1234E'
APP = str(ROOT / 'ui' / 'app.py')


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('UCRII_DATA_DIR', str(tmp_path))
    return tmp_path


def submit(pan):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input(key='pan_input').set_value(pan)
    next(b for b in at.button if b.label == 'Generate / Score').click()
    return at.run()


def test_valid_pan_scores(data_dir):
    at = submit(PAN)
    assert not at.exception
    assert any(m.label == 'UCRII' for m in at.metric)
    assert [t.label for t in at.tabs] == ['Score', 'Why this score', 'Activity and balance', 'Spending analytics', 'Credit signals', 'Data and method']
    assert (data_dir / f'{PAN}.csv').exists()
    assert not [w.value for w in at.warning if 'unavailable' in w.value]      # tabs swallow errors into warnings
    at.slider[0].set_value(3000).run()                                          # what-if reacts to the slider
    assert not at.exception and any(m.label == 'Months with a shortfall' for m in at.metric)


def test_invalid_pan_shows_error_and_writes_nothing(data_dir):
    at = submit('../x')
    assert not at.exception
    assert any('Invalid PAN' in e.value for e in at.error)
    assert list(data_dir.iterdir()) == []


def test_existing_pan_is_reused(data_dir):
    S.score_pan(PAN, data=data_dir)
    t0 = (data_dir / f'{PAN}.csv').stat().st_mtime_ns
    at = submit(PAN.lower())
    assert not at.exception
    assert any('Existing history found' in i.value for i in at.info)
    assert (data_dir / f'{PAN}.csv').stat().st_mtime_ns == t0
