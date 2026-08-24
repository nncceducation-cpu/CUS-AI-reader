# Model directory

No clinical model weights are included.

Place a validated ONNX feature model and one matching `*.manifest.json` file here. The app refuses inference when the model file is missing and visibly guards a manifest marked unvalidated.

The adapter expects:

- one grayscale tensor input shaped `[batch, 1, height, width]`
- pixel values scaled to 0 through 1
- one output shaped `[batch, number_of_labels]`
- one probability per label listed in the manifest
- mandatory `plane_coronal` and `plane_sagittal` outputs
- optional `plane_posterior_fossa` and `plane_other` outputs
- every decoded frame to be evaluated in sequential batches without diagnostic frame sampling

Recommended labels are feature-level outputs such as laterality, germinal matrix hemorrhage, intraventricular blood, ventricular distension, focal periventricular echogenicity, cysts, and cerebellar hemorrhage. The model should not directly emit the final consensus grade. Per-frame probabilities are aggregated within accepted planes, then clinician-verified features enter the Canadian consensus rule engine so every classification remains traceable.
