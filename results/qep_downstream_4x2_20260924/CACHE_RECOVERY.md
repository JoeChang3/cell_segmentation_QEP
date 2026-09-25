# Historical-cache recovery after failed Q2 reconstruction parity

The original MacBook run passed all Raw-P, Raw-C and FastGP-P anchor checks on
both development images. Eight non-QEP combinations completed. Newly fitted
nuclei Q2-C yielded AP50 0.5974358974 instead of 0.625, and stopped as designed.
This result remains a reproduction discrepancy, not a new scientific comparison.

User supplied original Round-3 and Round-5 caches for both q values and both
images. All four pairs of predmean arrays are pixel-identical. New nuclei Q2
vs historical Q2: MAE 0.0868011729, max absolute difference 0.9513549805 in raw
intensity units. Aggregate fitted lengthscale/outputscale/noise/final loss agree
to floating-point precision. These aggregates do not establish per-tile parameter
identity. The underlying prediction discrepancy is unresolved; no claim is made
that thread count, CG tolerance, or a specific solver has been proved responsible.

Recovery uses the original approved preference for historical reconstructions.
The added cache manifest binds the four user-supplied Round-3 files by SHA256,
array SHA256, dtype, shape and run identifier. The runner will score the historical
arrays under both P and C, and still require the complete Round-5 QEP-C anchor
checks. A mismatch still stops execution. No optimizer, model, downstream rule,
metric, or anchor tolerance changes. The failed run is retained in its original
output folder; the new run uses a separate folder.

Validation here: four authentic cache loads passed; wrong geometry and altered
bytes were rejected; Python syntax checks passed. Full P/C execution is pending
on the MacBook with the verified research environment. Cached reconstruction and
total runtimes are null, not zero. The historical full-pipeline times are saved
separately and are not presented as isolated reconstruction times.
