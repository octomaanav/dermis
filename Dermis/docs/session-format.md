# Dermis session package

`.dermis` is a directory package. It contains an append-only manifest and stream chunks so interrupted recordings can be recovered without rewriting prior data.

```text
Session.dermis/
  manifest.json
  streams/*.chunk
  streams/index.json
  events.jsonl
  annotations.json
  calibration/*.json
  video/*.mov
  audio/*.m4a
  checksums.json
```

Raw device values, status, flags, sequence numbers, device timestamps, and host timeline mappings are mandatory. Derived values are optional caches and must name the pipeline and calibration versions used.

