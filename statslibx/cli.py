import argparse
from statslibx.io import load_file
from statslibx.preprocessing import Preprocessing


def main():
    parser = argparse.ArgumentParser(
        prog="statslibx",
        description="Statslibx - Data analysis from terminal"
    )

    subparsers = parser.add_subparsers(dest="command")

    # describe
    describe = subparsers.add_parser("describe")
    describe.add_argument("file")

    # quality
    quality = subparsers.add_parser("quality")
    quality.add_argument("file")

    # preview
    preview = subparsers.add_parser("preview")
    preview.add_argument("file")
    preview.add_argument("-n", "--rows", type=int, default=5)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    df = load_file(args.file)
    pp = Preprocessing(df)

    if args.command == "describe":
        print(pp.describe_numeric())

    elif args.command == "quality":
        print(pp.data_quality())

    elif args.command == "preview":
        print(pp.preview_data(args.rows))


if __name__ == "__main__":
    main()
