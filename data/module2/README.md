# Module 2 evaluation data

`raw/` contains immutable upstream downloads; `images/`, `annotations/` and
`masks/` contain canonical converted evaluation inputs. The repository starts
with manifests and no newly downloaded data. Populate `sources.json`, prepare
the approved sources, then add reviewed JSONL cases before claiming any real
model quality gate.
