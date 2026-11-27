import os
import shutil
import subprocess
import sysconfig

import pytest


COMMANDS = ("api", "logs", "bump-versions", "run-migrations")
GUIDE_URL = "https://docs.braintrust.dev/admin/self-hosting/aws-cloudformation"


@pytest.fixture
def cli():
    executable = shutil.which("braintrust", path=sysconfig.get_path("scripts"))
    assert executable is not None
    return executable


@pytest.mark.parametrize("command", [(), *((name,) for name in COMMANDS)])
def test_install_help(cli, command):
    result = subprocess.run(
        [cli, "install", *command, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "deprecated" in result.stdout
    assert "Terraform" not in result.stdout
    assert GUIDE_URL in result.stdout
    if not command:
        assert result.stdout.count("(deprecated)") == len(COMMANDS)


@pytest.mark.parametrize("command", COMMANDS)
def test_install_warns_on_stderr_before_aws_setup(cli, command, tmp_path):
    result = subprocess.run(
        [cli, "install", command, "test-stack"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "AWS_CONFIG_FILE": str(tmp_path / "config"),
            "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "credentials"),
            "AWS_DEFAULT_PROFILE": "missing-test-profile",
            "AWS_PROFILE": "missing-test-profile",
            "AWS_EC2_METADATA_DISABLED": "true",
        },
    )

    assert result.returncode == 1
    warning = result.stderr.splitlines()[0]
    assert warning.startswith("Warning:")
    assert "deprecated" in warning
    assert "Terraform" not in warning
    assert GUIDE_URL in warning
    assert GUIDE_URL not in result.stdout


def test_install_requires_subcommand(cli):
    result = subprocess.run(
        [cli, "install"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Warning:" not in result.stderr
