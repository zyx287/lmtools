"""
Multi-scale analysis of FIB-SEM dataset
"""
import sys
import logging
import pkgutil
import importlib
import timeit
from argparse import ArgumentParser

import lmtools.cli as cli_package

logger = logging.getLogger(__name__)


def main(commandline_arguments=None) -> int:

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s: %(message)s")

    parser = ArgumentParser(description=__doc__, prog="lmtools")
    

    subparsers = parser.add_subparsers()

    modules = pkgutil.iter_modules(cli_package.__path__)

    for _, module_name, _ in modules:
        module = importlib.import_module("." + module_name, cli_package.__name__)
        help = module.__doc__.strip().split("\n", maxsplit=1)[0]
        subparser = subparsers.add_parser(
            module_name,
            help=help,
            description=module.__doc__
        )
        subparser.set_defaults(module=module)

        module.add_arguments(subparser)

    args = parser.parse_args(commandline_arguments)


    if not hasattr(args, "module"):
        parser.error("Provide the subcommand to run")
    else:
        module = args.module
        del args.module
        module_name = module.__name__.split('.')[-1]
        sys.stderr.write(f"SETTINGS FOR: {module_name} \n")
        for object_variable, value in vars(args).items():
            sys.stderr.write(f" {object_variable}: {value}\n")
        tic = timeit.default_timer()
        logger.info(f"Starting timer ({module_name})")
        module.main(args)
        toc = timeit.default_timer()
        processing_time = toc - tic
        logger.info(f"Elapsed time ({module_name}): {round(processing_time, 4)}s")
        logger.info("Done")
    return 0

if __name__ == "__main__":
    sys.exit(main())