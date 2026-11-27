from argparse import RawDescriptionHelpFormatter

from . import api, bump_versions, logs, run_migrations
from ._deprecation import DEPRECATION


def build_parser(subparsers, parent_parser):
    install_parser = subparsers.add_parser(
        "install",
        help="Set up and verify a Braintrust CloudFormation stack (deprecated)",
        description=DEPRECATION,
        formatter_class=RawDescriptionHelpFormatter,
        parents=[parent_parser],
    )
    install_subparsers = install_parser.add_subparsers(dest="install_subcommand", required=True)

    for module in [api, logs, bump_versions, run_migrations]:
        module.build_parser(install_subparsers, parents=[parent_parser])
