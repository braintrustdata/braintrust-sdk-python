"""
Compatibility tests for the Anthropic wrapper import path.
"""

from braintrust.wrappers.test_utils import run_in_subprocess, verify_autoinstrument_script


class TestAnthropicWrapperCompat:
    def test_anthropic_wrapper_compat_exports(self):
        result = run_in_subprocess("""
            from braintrust.wrappers.anthropic import wrap_anthropic as compat_wrap
            from braintrust.integrations.anthropic import wrap_anthropic as new_wrap
            from braintrust.integrations.anthropic import wrap_anthropic_client

            assert compat_wrap is new_wrap
            assert callable(wrap_anthropic_client)
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_anthropic_integration_setup_wraps_supported_clients(self):
        result = run_in_subprocess("""
            from braintrust.integrations.anthropic import AnthropicIntegration
            import anthropic

            original_sync_module = type(anthropic.Anthropic(api_key="test-key").messages).__module__
            original_async_module = type(anthropic.AsyncAnthropic(api_key="test-key").messages).__module__
            AnthropicIntegration.setup()
            patched_sync = anthropic.Anthropic(api_key="test-key")
            patched_async = anthropic.AsyncAnthropic(api_key="test-key")

            assert type(patched_sync.messages).__module__ == "braintrust.integrations.anthropic.tracing"
            assert type(patched_async.messages).__module__ == "braintrust.integrations.anthropic.tracing"
            assert type(patched_sync.messages).__module__ != original_sync_module
            assert type(patched_async.messages).__module__ != original_async_module
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_anthropic_integration_setup_is_idempotent(self):
        result = run_in_subprocess("""
            import inspect
            from braintrust.integrations.anthropic import AnthropicIntegration
            import anthropic

            AnthropicIntegration.setup()
            first_sync_init = inspect.getattr_static(anthropic.Anthropic, "__init__")
            first_async_init = inspect.getattr_static(anthropic.AsyncAnthropic, "__init__")

            AnthropicIntegration.setup()

            assert inspect.getattr_static(anthropic.Anthropic, "__init__") is first_sync_init
            assert inspect.getattr_static(anthropic.AsyncAnthropic, "__init__") is first_async_init
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout

    def test_anthropic_integration_setup_can_disable_specific_patchers(self):
        result = run_in_subprocess("""
            from braintrust.integrations.anthropic import AnthropicIntegration
            import anthropic

            AnthropicIntegration.setup(disabled_patchers={"anthropic.init.async"})
            patched_sync = anthropic.Anthropic(api_key="test-key")
            unpatched_async = anthropic.AsyncAnthropic(api_key="test-key")

            assert type(patched_sync.messages).__module__ == "braintrust.integrations.anthropic.tracing"
            assert type(unpatched_async.messages).__module__.startswith("anthropic.")
            print("SUCCESS")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout


class TestAutoInstrumentAnthropic:
    """Tests for auto_instrument() with Anthropic."""

    def test_auto_instrument_anthropic(self):
        """Test auto_instrument patches Anthropic, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_anthropic.py")

    def test_auto_instrument_anthropic_patch_config(self):
        verify_autoinstrument_script("test_auto_anthropic_patch_config.py")

    def test_auto_instrument_rejects_non_bool_option_for_openai(self):
        result = run_in_subprocess("""
            from braintrust.auto import auto_instrument
            from braintrust.integrations import IntegrationPatchConfig

            try:
                auto_instrument(openai=IntegrationPatchConfig())
            except TypeError as exc:
                assert "must be a bool" in str(exc)
                print("SUCCESS")
            else:
                raise AssertionError("Expected TypeError")
        """)
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout
