"""Historical average: the training mean for each sensor, weekday and 5-minute slot.

Persistence gets weak at longer horizons because traffic repeats every day and
week. This uses no recent readings, so a model that beats it learned more than
the calendar.
"""

import numpy as np
import pandas as pd
import torch

from .model import Metrics

SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


def week_slots(times):
    """Index of each timestamp's five-minute slot within a Monday-first week."""
    times = pd.DatetimeIndex(times)
    minute = times.hour.to_numpy() * 60 + times.minute.to_numpy()
    return (times.dayofweek.to_numpy() * SLOTS_PER_DAY + minute // 5).astype(np.int64)


class HistoricalAverage:
    def fit(self, times, values, mask):
        """Fit from observed training readings only (``mask`` marks observed)."""
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.shape != mask.shape or len(values) != len(times):
            raise ValueError("Times, values, and mask must align")
        slots = week_slots(times)
        sensors = values.shape[1]
        total = np.zeros((SLOTS_PER_WEEK, sensors))
        count = np.zeros((SLOTS_PER_WEEK, sensors))
        np.add.at(total, slots, np.where(mask, values, 0.0))
        np.add.at(count, slots, mask.astype(np.float64))
        if (count.sum(axis=0) == 0).any():
            raise ValueError("A sensor has no observed training values")
        overall = total.sum(axis=0) / count.sum(axis=0)
        # Slots never observed on a given weekday fall back to the same time of
        # day pooled across weekdays, then to the sensor's overall mean.
        daily_total = total.reshape(7, SLOTS_PER_DAY, sensors).sum(axis=0)
        daily_count = count.reshape(7, SLOTS_PER_DAY, sensors).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            by_day_time = np.where(daily_count > 0, daily_total / daily_count, np.nan)
            table = np.where(count > 0, total / count, np.nan)
        pooled = np.tile(by_day_time, (7, 1, 1)).reshape(SLOTS_PER_WEEK, sensors)
        table = np.where(np.isnan(table), pooled, table)
        self.table = np.where(np.isnan(table), overall, table).astype(np.float32)
        self.observed_slots = int((count > 0).sum())
        return self

    def predict(self, times):
        return self.table[week_slots(times)]


def score_table(dataset, predicted, daily=None, chunk=2048):
    """Score a per-row prediction table on a Windows dataset's targets.

    predicted[r] is the forecast for row r of the dataset's split. It goes through
    the same Metrics as the neural models, so the numbers compare directly.
    """
    if predicted.shape != dataset.y.shape:
        raise ValueError("Prediction table must match the split's rows and sensors")
    metrics = Metrics(dataset.horizon)
    steps = np.arange(dataset.horizon)
    for start in range(0, len(dataset), chunk):
        origin = np.arange(start, min(start + chunk, len(dataset)))
        rows = origin[:, None] + dataset.history + steps[None, :]
        prediction = torch.from_numpy(predicted[rows])
        target = torch.from_numpy(dataset.y[rows])
        mask = torch.from_numpy(dataset.mask[rows])
        metrics.update(prediction, target, mask)
        if daily is not None:
            daily.update(prediction, target, mask, start)
    return metrics.result()
