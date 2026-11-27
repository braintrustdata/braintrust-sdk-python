import sys
import textwrap
from argparse import RawDescriptionHelpFormatter
from functools import wraps


DEPRECATION = (
    "The braintrust install commands are deprecated.\n"
    "Use the CloudFormation guide for existing stacks:\n"
    "https://docs.braintrust.dev/admin/self-hosting/aws-cloudformation"
)


def add_deprecated_parser(subparsers, name, summary, parents, handler):
    parser = subparsers.add_parser(
        name,
        help=f"{summary} (deprecated)",
        description=f"{summary}.\n\n{DEPRECATION}",
        formatter_class=RawDescriptionHelpFormatter,
        parents=parents,
    )

    @wraps(handler)
    def run(args):
        print("Warning: " + DEPRECATION.replace("\n", " "), file=sys.stderr)
        try:
            return handler(args)
        except ModuleNotFoundError as exc:
            if exc.name not in {"boto3", "botocore"}:
                raise
            raise ModuleNotFoundError(
                textwrap.dedent(
                    f"""\
                    At least one dependency not found: {str(exc)!r}
                    It is possible that braintrust was installed without the CLI dependencies. Run:

                      pip install 'braintrust[cli]'

                    to install braintrust with the CLI dependencies (make sure to quote 'braintrust[cli]')."""
                )
            ) from exc

    parser.set_defaults(func=run)
    return parser
