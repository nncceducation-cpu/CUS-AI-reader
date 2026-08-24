# Model directory

No clinical model weights are included.

Place a validated ONNX feature model and one matching `*.manifest.json` file here. The app refuses inference when the model file is missing and visibly guards a manifest marked unvalidated.

The adapter expects:

- one grayscale tensor input shaped `[batch, 1, height, width]`
- pixel values scaled to 0 through 1
- one output shaped `[batch, number_of_labels]`
- one probability per label listed in the manifest

Recommended labels are feature-level outputs such as plane, laterality, germinal matrix hemorrhage, intraventricular blood, ventricular distension, focal periventricular echogenicity, cysts, and cerebellar hemorrhage. The model should not directly emit the final consensus grade. The rule engine owns that mapping so every classification remains traceable.

