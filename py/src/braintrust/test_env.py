from .env import BraintrustEnv, EnvParser, EnvVar, parse_bool, parse_float, parse_int, parse_string


class TestEnvParsers:
    def test_parse_float(self):
        assert parse_float("123.45") == 123.45
        assert parse_float("nan") is None
        assert parse_float("inf") is None
        assert parse_float("") is None
        assert parse_float("not_a_number") is None

    def test_parse_int(self):
        assert parse_int("123") == 123
        assert parse_int("-5") == -5
        assert parse_int("") is None
        assert parse_int("1.2") is None
        assert parse_int("not_an_int") is None

    def test_parse_bool(self):
        for value in ("true", "True", "1", "yes", "y", "on"):
            assert parse_bool(value) is True
        for value in ("false", "False", "0", "no", "n", "off"):
            assert parse_bool(value) is False
        assert parse_bool("") is None
        assert parse_bool("maybe") is None

    def test_parse_string(self):
        assert parse_string("value") == "value"
        assert parse_string("") is None


class TestEnvVar:
    def test_returns_default_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_ENV_VAR", raising=False)
        assert EnvVar("TEST_ENV_VAR", EnvParser.INT).get(42) == 42

    def test_returns_default_when_env_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_ENV_VAR", "invalid")
        assert EnvVar("TEST_ENV_VAR", EnvParser.INT).get(42) == 42

    def test_reads_environment_lazily(self, monkeypatch):
        env_var = EnvVar("TEST_ENV_VAR", EnvParser.INT)
        monkeypatch.setenv("TEST_ENV_VAR", "1")
        assert env_var.get(42) == 1
        monkeypatch.setenv("TEST_ENV_VAR", "2")
        assert env_var.get(42) == 2

    def test_default_is_supplied_by_call_site(self, monkeypatch):
        env_var = EnvVar("TEST_ENV_VAR", EnvParser.INT)
        monkeypatch.delenv("TEST_ENV_VAR", raising=False)
        assert env_var.get(1) == 1
        assert env_var.get(2) == 2


class TestBraintrustEnv:
    def test_centralized_env_definitions_are_lazy(self, monkeypatch):
        monkeypatch.delenv("BRAINTRUST_HTTP_TIMEOUT", raising=False)
        assert BraintrustEnv.HTTP_TIMEOUT.get(60.0) == 60.0
        monkeypatch.setenv("BRAINTRUST_HTTP_TIMEOUT", "0.2")
        assert BraintrustEnv.HTTP_TIMEOUT.get(60.0) == 0.2

    def test_otel_compat_uses_shared_bool_parser(self, monkeypatch):
        for value in ("true", "1", "yes"):
            monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", value)
            assert BraintrustEnv.OTEL_COMPAT.get(False) is True

        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "false")
        assert BraintrustEnv.OTEL_COMPAT.get(True) is False
