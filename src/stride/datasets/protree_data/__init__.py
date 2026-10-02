try:
    import click
except ImportError:
    click = None

from stride.datasets.protree_data.static import download_all, DEFAULT_DATA_DIR


def _cli_entry():
    if click is None:
        raise ImportError("Dataset download CLI requires 'click'. Install with: pip install click")

    @click.command()
    @click.option("--directory", "-d", default=DEFAULT_DATA_DIR, help="Directory to store datasets")
    @click.option("--silent", "-s", is_flag=True, help="Suppress displaying progress.")
    @click.option(
        "--dataset-names",
        "-n",
        default="all",
        help="Comma-separated list of dataset names to download. "
        "Allowable values are 'breast_cancer', 'caltech', 'compass', "
        "'diabetes', 'mnist' and 'rhc'. Use 'all' to download all "
        "datasets.",
    )
    def main(directory, silent, dataset_names):
        download_all(directory=directory, dataset_names=[s.strip() for s in dataset_names.split(",")], verbose=not silent)

    return main()


if __name__ == "__main__":
    _cli_entry()
