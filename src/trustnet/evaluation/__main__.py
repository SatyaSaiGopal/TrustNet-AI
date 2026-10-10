"""Run the synthetic benchmark: ``python -m trustnet.evaluation``."""

import argparse

from trustnet.evaluation.benchmark import format_report, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m trustnet.evaluation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--development-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    report = run_benchmark(args.seed, args.development_size, args.test_size)
    print(report.model_dump_json(indent=2) if args.json else format_report(report))


if __name__ == "__main__":
    main()
