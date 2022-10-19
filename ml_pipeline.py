"""
This script executes the ML training pipeline
"""
import argparse
import src.cleaning_data as basic_cleaning
import src.train_model as train_test_model
import src.slice_performance as check_score
import logging


def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "basic_cleaning":
        logging.info("Basic cleaning procedure started")
        basic_cleaning.execute_cleaning()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("Train/Test model procedure started")
        train_test_model.train_test_model()

    if args.action == "all" or args.action == "check_score_slices":
        logging.info("Performance in slices check procedure started")
        check_score.check_score_slices()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["basic_cleaning",
                 "train_test_model",
                 "check_score_slices",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    execute(main_args)
