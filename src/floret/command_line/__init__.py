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
        "--position_min",
        dest="position_min",
        type=int,
        default=0,
        help="The minimum normalised position.",
    )
    parser.add_argument(
        "--position_max",
        dest="position_max",
        type=int,
        default=1,
        help="The maximum normalised position.",
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=["spiral", "symmetric", "swinging"],
        default="symmetric",
        help="Do a symmetric, spiral or swinging scheme",
    )
    parser.add_argument(
        "--nhelix",
        dest="nhelix",
        type=int,
        default=1,
        help="The nhelix order.",
    )
    parser.add_argument(
        "--order_by",
        dest="order_by",
        type=str,
        default="angle",
        choices=["angle", "position"],
        help=" ".join(
            [
                "If order_by=position, then for each position, all angles are collected.",
                "If order=angle, then for each angle all positions are collected",
            ]
        ),
    )
    parser.add_argument(
        "--interleave_positions",
        dest="interleave_positions",
        type=bool,
        default=True,
        help="Skip adjacent positions and interleave if order_by=angle",
    )

    # Skipnum and symmetry are mutually exclusive
    parameter_group = parser.add_mutually_exclusive_group()
    parameter_group.add_argument(
        "--symmetry",
        dest="symmetry",
        type=int,
        default=0,
        help="The scan symmetry order.",
    )
    parameter_group.add_argument(
        "--stepnum",
        dest="stepnum",
        type=int,
        default=0,
        help="The number of images to step (sprial and swinging).",
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
    positions, angles = floret.generate_scan(
        tilt_angle_zero=args.tilt_angle_zero,
        tilt_angle_min=args.tilt_angle_min,
        tilt_angle_max=args.tilt_angle_max,
        tilt_angle_step=args.tilt_angle_step,
        num_tilt_angles=args.num_tilt_angles,
        mode=args.mode,
        symmetry=args.symmetry,
        stepnum=args.stepnum,
        nhelix=args.nhelix,
        position_min=args.position_min,
        position_max=args.position_max,
        order_by=args.order_by,
        interleave_positions=args.interleave_positions,
    )

    # Print the angles
    for p, a in zip(positions, angles):
        logging.info("%f, %f" % (p, a))


def main(args: List[str] = None):
    """
    Do the alignment

    """
    main_impl(get_parser().parse_args(args=args))
