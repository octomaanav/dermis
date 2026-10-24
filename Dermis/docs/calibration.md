# Calibration model

Calibration profiles are immutable, versioned JSON documents tied to a pod ID and skin ID. Sessions store the profile identifier and a snapshot. Reprocessing never mutates the original raw stream.

No UI or export may label a value in newtons unless its sample was produced by a valid, non-stale force calibration profile.

