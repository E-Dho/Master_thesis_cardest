# POL MobilityDB Loader

This project-local loader builds MobilityDB without sudo and loads the enriched POL
trajectory dataset into PostgreSQL/PostGIS/MobilityDB tables.

Cluster paths:

- Scripts: `/zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader`
- Env: `/work_beegfs/sunip956/master_thesis_trajectories/envs/mobilitydb`
- DB root: `/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb`
- Dataset: `/work_beegfs/sunip956/master_thesis_trajectories/datasets/pol_atl_1000a_365d_5min_seed1_enriched_traj_20260714T111045Z`

Typical workflow:

```bash
sbatch /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/build_mobilitydb.sbatch
/zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/init_mobilitydb.sh
sbatch /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/load_pol_to_mobilitydb.sbatch
```

The reusable database is stored in work storage, but PostgreSQL should be run only
inside a Slurm job or interactive allocation. The load job starts the server,
loads and validates the data, and stops the server at the end.

For an explicit reusable server allocation:

```bash
sbatch /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/start_mobilitydb.sbatch
source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh
module load micromamba/1.4.2
export MAMBA_ROOT_PREFIX=/work_beegfs/sunip956/micromamba
micromamba run -p /work_beegfs/sunip956/master_thesis_trajectories/envs/mobilitydb \
  psql -h /work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/run -p 55432 -d pol_mobilitydb
```

Stop it with:

```bash
/zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/stop_mobilitydb.sh
```

Tables:

- `pol.agents(agent_id, age, "educationLevel", interest, joviality, family_size)`
- `pol.trips(trip_id, agent_id, start_time, end_time, num_of_segments, trip_tgeom, trip_geom)`
- `pol.segments(trip_id, segment_idx, s_x, s_y, e_x, e_y, t_s, t_e, segment_tgeom, segment_geom)`

Trip IDs use `trip_id = agent_id * B + per_agent_trip_index`, where
`B = 10^ceil(log10(m + 1))` and `m` is the maximum per-agent trip count.
