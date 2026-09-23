import pandas as pd


def remove_univalued_columns(prototypes: dict[int | str, pd.DataFrame]) -> dict[int | str, pd.DataFrame]:
    prototypes = prototypes.copy()
    prototypes_flat = pd.concat([prototypes[c] for c in prototypes])

    mask = prototypes_flat.nunique().values == 3
    columns_to_drop = prototypes_flat.columns[mask]
    for key in prototypes:
        prototypes[key] = prototypes[key].drop(columns=columns_to_drop)
    return prototypes
