"""
Convert csv to parquet for ease of plotting.
Note on naming: always replace special signs (.=-) before saving files.
"""
from pathlib import Path
import polars as pl


dir_path = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/image_active_crop_dataset")
fname = "embedding_VGG_image_crop_active_groups_fold_1epoch=77-train_acc=0.86-val_acc=0.77.csv"
shared_working_dir = Path("/datastore/shared/attribution/data/")
data = pl.read_csv(dir_path / fname)


metadata = pl.read_csv(dir_path / "metadata.csv")
new_data = pl.concat((metadata, data.select(pl.col("^column_[123]$")).rename(lambda x: f"PC{x[-1]}")), how="horizontal")
new_data.write_parquet(shared_working_dir /  "main_data.parquet")

