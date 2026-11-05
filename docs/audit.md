# Audit

Before building on the code I went through it looking for leakage, wrong assumptions and things that would break at a bigger scale. This page says what I checked, what I found and what is still open. Nothing in here is a result of the models; the numbers about the data come from `tools/data_audit.py` ([data-audit.json](../reports/data-audit.json)).

## Bugs found and fixed

- **The sensor-subset code couldn't do all 207 sensors.** It walked the graph from the first sensor and raised an error if that walk didn't reach enough sensors. The graph has 2 components, so asking for all of them failed. The new `order()` finishes the first component and then adds the leftover sensors in file order, and every subset is a prefix of that one list. The old rule is the same walk, so the 16-sensor subset is unchanged (a test checks it against the old function, and the seed-42 benchmark runs still give the recorded model hashes).
- **The dense graph ablation could not isolate the graph.** The "self only" model still got every sensor's readings through its input layer, so it was never really without neighbors. The earlier write-up said so, but the experiment couldn't fix it. The shared-weight models can: a shared GRU sees only its own sensor, and the shared Graph-GRU adds the two neighbor averages, nothing else.
- **A plotting crash after a long run** (found earlier): early-stopped runs have different lengths and the convergence plot assumed equal ones. Fixed, with a test.

I didn't find anything else that was wrong in the original pipeline, and I'd rather say that than pad this list.

## Leakage checks

None found. These are checked by tests, not just by reading the code (`tests/test_leakage.py` and the tests next to each module):

- Rewrite every test row and the trained weights, the validation curve and the chosen epoch stay bit-for-bit the same. Only the test score changes.
- Rewrite the calibration rows (the part used for conformal intervals) and training and checkpoint choice don't move either.
- The scalers, the training windows, the historical average and the calendar features come from training rows or from the clock only.
- The graph comes from published distances and never from speeds, and the chosen sensors don't depend on speeds.
- In the day-by-day replay a window counts as known only once its last target row has been observed. The tests corrupt the not-yet-arrived targets and check that no strategy's forecasts change, and that the online interval methods never read an outcome before it arrives.

One choice worth knowing about: in the replay experiment, missing readings are forward-filled across the split boundaries (the past is known when a model runs for real). The other experiments fill within each split, which is slightly more conservative at the start of validation and test. It only affects the first few windows of a split.

## What the data looks like

- **Graph:** 1515 directed edges, 2 components (sizes 206 and 1), and the sensor order in the data file matches the graph file.
- **Timestamps:** the slowest weekday hours are 17:00, 16:00, 8:00 (the evening and morning rush) and weekends are slowest around 13:00, 14:00, so the timestamps behave like local Los Angeles time even though the file doesn't say.
- **Missing readings** (NaN or zero) are 7.1% of the readings in the training period and 7.0% in validation but 12.1% in the test period, and 7 sensors have more than 30% missing in the test period against none in training. So the test period isn't just later, it is noisier. Every experiment masks missing targets, so scores only cover observed readings, but a model sees more stale (forward-filled) inputs in the test period than it was trained on.
- **Congestion:** 10.5% of observed training readings are below 40 mph, 12.3% in the test period.
- **The first 16 sensors** sit on one highway corridor ([map](../reports/scaling/scaling-subsets.png)), which is a limit on what the earlier results say about the whole network.

## Limits that are still open

- Zero counts as missing, which also drops real stopped traffic.
- One city and one four-month period, with one test window of 24 days.
- Overlapping windows (one per five minutes) are strongly correlated, so the number of independent test points is far smaller than the number of windows. The test-day intervals are my way of respecting that, and they are wide for that reason.
- Seeds: two to five per cell depending on the experiment.

## Speed and memory

Nothing in the data pipeline was slow enough to change. The dense models cost about the same at every sensor count. The shared models do one recurrent pass per sensor and window, so their time per update grows roughly with the sensor count; that cost is measured in the [scaling report](../reports/scaling/report.md).
