from braintrust import version


def test_version():
    """Test the version module from source. Templating is checked against the wheel in tests/wheel."""
    assert version.VERSION
    assert isinstance(version.VERSION, str)
    assert version.GIT_COMMIT == "__GIT_COMMIT__"
    assert version.RELEASE_CHANNEL == "source"
