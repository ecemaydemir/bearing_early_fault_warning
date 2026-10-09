from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
if not (ROOT / "reports" / "scores.csv.gz").exists():
    pytest.skip("run `make experiment` first", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.parametrize("test", ["test1", "test2", "test3"])
@pytest.mark.parametrize("detector", ["RMS threshold", "LSTM autoencoder"])
def test_dashboard_renders(test, detector):
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    at.sidebar.selectbox[0].select(test).run()
    at.sidebar.selectbox[1].select(detector).run()
    at.sidebar.slider[0].set_value(at.sidebar.slider[0].max).run()
    assert not at.exception
    assert any("ALARM" in m.value for m in at.markdown)
