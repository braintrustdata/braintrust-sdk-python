import logging

from ._deprecation import add_deprecated_parser


_logger = logging.getLogger("braintrust.install.logs")


def build_parser(subparsers, parents):
    parser = add_deprecated_parser(subparsers, "run-migrations", "Run schema migrations", parents, main)
    parser.add_argument("name", help="Name of the CloudFormation stack")


def main(args):
    # pylint: disable-next=no-name-in-module
    from ...aws import LazyClient, cloudformation

    resources = cloudformation.describe_stack_resources(StackName=args.name)
    migration_lambda = [r for r in resources["StackResources"] if r["LogicalResourceId"] == "MigrateDatabaseFunction"]
    if not migration_lambda:
        raise ValueError("No MigrateDatabaseFunction found in the stack.")
    migration_lambda = migration_lambda[0]
    arn = migration_lambda["PhysicalResourceId"]
    lambda_ = LazyClient("lambda")
    lambda_.invoke(FunctionName=arn)
