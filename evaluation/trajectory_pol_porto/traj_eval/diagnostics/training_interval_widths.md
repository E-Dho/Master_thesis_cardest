# POL Training Interval Widths

## Training contexts

| Interval kind | Two-sided contexts | Share of all contexts | Projected two-sided (20k steps) | One-sided | Median width | Width bins |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| segment_ordinary | 11683 | 2.85% | 1168300 | 35607 | 1.62e+07 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 0 (0%), 1 h-1 day: 3 (0.0257%), 1-30 days: 299 (2.56%), 30-180 days: 5215 (44.6%), > 180 days: 6166 (52.8%) |
| segment_trajectory_temporal | 39681 | 9.69% | 3968100 | 0 | 1.61e+07 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 0 (0%), 1 h-1 day: 4 (0.0101%), 1-30 days: 1092 (2.75%), 30-180 days: 17803 (44.9%), > 180 days: 20782 (52.4%) |
| trip_ordinary | 32755 | 8% | 3275500 | 98217 | 1.62e+07 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 0 (0%), 1 h-1 day: 2 (0.00611%), 1-30 days: 897 (2.74%), 30-180 days: 14578 (44.5%), > 180 days: 17278 (52.7%) |

### Equality predicates in training contexts

| Column | Count | Share of all contexts |
| --- | ---: | ---: |
| segments:t_e | 14927 | 3.64% |
| segments:t_s | 14717 | 3.59% |
| trips:end_time | 40778 | 9.96% |
| trips:start_time | 41201 | 10.1% |

## POL v3 benchmark core predicates

| Interval kind/category | Two-sided predicates | Share of all two-sided predicates | Median width | Width bins |
| --- | ---: | ---: | ---: | --- |
| segment_time/spatio_temporal | 1131 | 53.1% | 8.83e+04 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 290 (25.6%), 1 h-1 day: 274 (24.2%), 1-30 days: 314 (27.8%), 30-180 days: 168 (14.9%), > 180 days: 85 (7.52%) |
| segment_time/temporal | 394 | 18.5% | 5.98e+04 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 119 (30.2%), 1 h-1 day: 86 (21.8%), 1-30 days: 116 (29.4%), 30-180 days: 56 (14.2%), > 180 days: 17 (4.31%) |
| trip_time/temporal | 606 | 28.4% | 6.63e+04 s | inside (U < L): 0 (0%), < 1 min: 0 (0%), 1 min-1 h: 183 (30.2%), 1 h-1 day: 132 (21.8%), 1-30 days: 184 (30.4%), 30-180 days: 80 (13.2%), > 180 days: 27 (4.46%) |
