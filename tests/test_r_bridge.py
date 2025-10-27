import pytest, shutil

@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available in CI environment")
def test_r_bridge_stub_import():
    # Just ensure module imports; functional test requires R packages
    import spectral_predict.readers.asd_r_bridge as rb
    assert hasattr(rb, "read_asd_via_r")
