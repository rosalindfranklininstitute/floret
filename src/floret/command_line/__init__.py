#
# Copyright (C) 2020 RFI
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import floret
import logging
from argparse import ArgumentParser
from typing import List


__all__ = ["main"]


# Get the logger
logger = logging.getLogger()


def get_description():
    """
    Get the program description

    """
    return "Do some analysis"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(prog="floret", description=get_description())

    # Add arguments
    parser.add_argument(
        "--tilt_angle_zero",
        dest="tilt_angle_zero",
        type=float,
        default=0,
        help="The zero tilt angle offset (degrees).",
    )
    parser.add_argument(
        "--tilt_angle_min",
        dest="tilt_angle_min",
        type=float,
        default=-90,
        help="The minimum tilt angle (degrees).",
    )
    parser.add_argument(
        "--tilt_angle_max",
        dest="tilt_angle_max",
        type=float,
        default=90,
        help="The maximum tilt angle (degrees).",
    )
    parser.add_argument(
        "--tilt_angle_step",
        dest="tilt_angle_step",
        type=float,
        default=None,
        help="The tilt angle step (degrees).",
    )
    parser.add_argument(
        "--num_tilt_angles",
        dest="num_tilt_angles",
        type=int,
        default=None,
        help="The number of tilt angles.",
    )
    parser.add_argument(
        "--symmetry",
        dest="symmetry",
        type=int,
        default=0,
        help="The scan symmetry order.",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=["spiral", "symmetric", "swing"],
        default="symmetric",
        help="Do a symmetric, spiral or swing scheme",
    )

    # Return the parser
    return parser


def main_impl(args):
    """
    Do the analysis

    """

    # Set the logger
    logging.basicConfig(level=logging.INFO, format="%(msg)s")

    # Generate scan angles
    angles = floret.generate_scan(
        tilt_angle_zero=args.tilt_angle_zero,
        tilt_angle_min=args.tilt_angle_min,
        tilt_angle_max=args.tilt_angle_max,
        tilt_angle_step=args.tilt_angle_step,
        num_tilt_angles=args.num_tilt_angles,
        symmetry=args.symmetry,
        mode=args.mode,
    )

    # Print the angles
    for a in angles:
        logging.info(a)


def main(args: List[str] = None):
    """
    Do the alignment

    """
    main_impl(get_parser().parse_args(args=args))
