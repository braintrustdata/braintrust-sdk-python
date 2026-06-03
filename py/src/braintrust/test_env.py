import pytest

from .env import (
    BraintrustEnv,
    EnvParser,
    EnvVar,
    parse_bool,
    parse_float,
    parse_int,
    parse_string,
)


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
        assert parse_string("   ") is None


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
    def test_api_key_nonblank_environment_wins(self, tmp_path, monkeypatch):
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=file-key\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BRAINTRUST_API_KEY", "env-key")

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) == "env-key"

    def test_api_key_blank_environment_falls_back_to_file(self, tmp_path, monkeypatch):
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=file-key\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BRAINTRUST_API_KEY", "   ")

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) == "file-key"

    def test_api_key_uses_nearest_parent_file(self, tmp_path, monkeypatch):
        nested = tmp_path / "packages" / "app"
        nested.mkdir(parents=True)
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=root-key\n")
        (tmp_path / "packages" / ".env.braintrust").write_text("BRAINTRUST_API_KEY=package-key\n")
        monkeypatch.chdir(nested)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) == "package-key"

    @pytest.mark.parametrize("contents", ["OTHER=value\n", 'BRAINTRUST_API_KEY="   "\n'])
    def test_api_key_nearest_file_is_boundary_without_nonblank_key(self, tmp_path, monkeypatch, contents):
        nested = tmp_path / "packages" / "app"
        nested.mkdir(parents=True)
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=root-key\n")
        (tmp_path / "packages" / ".env.braintrust").write_text(contents)
        monkeypatch.chdir(nested)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) is None

    def test_api_key_unreadable_nearest_file_is_boundary(self, tmp_path, monkeypatch):
        nested = tmp_path / "packages" / "app"
        nested.mkdir(parents=True)
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=root-key\n")
        (tmp_path / "packages" / ".env.braintrust").mkdir()
        monkeypatch.chdir(nested)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) is None

    def test_api_key_searches_cwd_plus_64_parents(self, tmp_path, monkeypatch):
        segments = [f"d{i}" for i in range(65)]
        nested = tmp_path.joinpath(*segments)
        nested.mkdir(parents=True)
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=too-high\n")
        monkeypatch.chdir(nested)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) is None

        (tmp_path / segments[0] / ".env.braintrust").write_text("BRAINTRUST_API_KEY=boundary-key\n")

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) == "boundary-key"

    def test_api_key_supports_dotenv_syntax_and_does_not_mutate_environment(self, tmp_path, monkeypatch):
        (tmp_path / ".env.braintrust").write_text('export BRAINTRUST_API_KEY="quoted-key" # comment\nOTHER=value\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
        monkeypatch.delenv("OTHER", raising=False)

        assert BraintrustEnv.API_KEY.get(None, use_dotenv=True) == "quoted-key"
        assert EnvVar("BRAINTRUST_API_KEY", EnvParser.STRING).get(None) is None
        assert EnvVar("OTHER", EnvParser.STRING).get(None) is None

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


class TestIdConfig:
    # LEGACY_IDS is lazily resolved from the current environment on each
    # access, like the other settings, so tests just set env vars and read it.

    def test_hex_ids_default(self, monkeypatch):
        monkeypatch.delenv("BRAINTRUST_OTEL_COMPAT", raising=False)
        monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
        assert BraintrustEnv.LEGACY_IDS is False

    def test_legacy_uuid_opt_out(self, monkeypatch):
        monkeypatch.delenv("BRAINTRUST_OTEL_COMPAT", raising=False)
        monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "true")
        assert BraintrustEnv.LEGACY_IDS is True

    def test_otel_compat_forces_hex(self, monkeypatch):
        # OTEL_COMPAT implies hex IDs regardless of LEGACY_IDS being unset.
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
        monkeypatch.delenv("BRAINTRUST_LEGACY_IDS", raising=False)
        assert BraintrustEnv.LEGACY_IDS is False

    def test_conflicting_flags_otel_wins(self, monkeypatch):
        # OTEL_COMPAT wins over LEGACY_IDS: legacy is disabled (hex IDs)
        # rather than raising.
        monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
        monkeypatch.setenv("BRAINTRUST_LEGACY_IDS", "true")
        assert BraintrustEnv.LEGACY_IDS is False
