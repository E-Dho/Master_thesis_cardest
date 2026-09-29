# Porto Taxi Loader

This directory creates an isolated MobilityDB benchmark from the Kaggle Porto taxi `train.csv`.

Place the unmodified source file at:

`/work_beegfs/sunip956/master_thesis_trajectories/datasets/porto_taxi/raw/train.csv`

Submit `load_porto_to_mobilitydb.sbatch`. It retains only `MISSING_DATA=false` traces with two or more valid WGS84 points, deterministically selects full trips up to 50,000,000 segments, loads EPSG:3763 geometry into `porto_taxi_mobilitydb` on port `55433`, validates the database, and writes the runtime query config.
