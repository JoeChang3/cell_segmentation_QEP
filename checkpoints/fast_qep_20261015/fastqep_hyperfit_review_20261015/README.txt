Fast-QEP hyperparameter learning: stage-one verification package
15 October 2026

For the finite-tile, covariance-matched qED prior with independent Gaussian noise, we derived the marginal-likelihood gradient, implemented Kronecker gradient calculations and passed 15 numerical checks. q was fixed at 1.5 or 2. This stage produced no new real-image experiments or AP measurements.

Package contents
- FastQEP_hyperparameter_feasibility_20261015.pdf gives the derivations, results, comparison design and criteria for proceeding.
- verify_hyperfit.py independently checks the mathematics. Use the original runner for real-image experiments.
- validation.json records the results, tolerances, environment and code SHA-256.
- source_provenance.json identifies the original R sources, frozen audits and supplied outputs.
- requirements.txt lists dependencies for the verification script. The working research environment can be reused.
- make_report.py generates the PDF with ReportLab and DejaVu fonts.
- SHA256SUMS.txt lists checksums for the delivered files.

Optional verification in the existing gpytorch_arm environment:

  conda activate gpytorch_arm
  cd /path/to/fastqep_hyperfit_review_20261015
  python verify_hyperfit.py --output validation_local.json

This typically takes seconds and requires neither cell images nor R or segmentation. Retain the supplied validation.json and keep the package separate from the original FastQEP_R_pipeline directory.

The checks support a computationally bounded implementation pilot. Demonstrating an advantage over GP requires a refitted Fast-GP control and a GP evaluated at the parameters learned by QEP. These controls distinguish optimization effects from model effects. Marginal-likelihood optimization on real images, sampling-gradient stability and runtime still needed testing at this stage.

The next step at the time of review was to obtain FastQEP_R_pipeline_20261015.zip from the Mac. The review outputs did not include the original sampler source, so integration with the runner could not yet be verified. Once recovered, the sampler and numerical safeguards needed review before fixing parameter ranges and acceptance rules. First-tile real-image tests and whole-image segmentation had not yet started.
