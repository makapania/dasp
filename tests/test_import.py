def test_import_package():
    import spectral_predict
    assert hasattr(spectral_predict, "VERSION")
