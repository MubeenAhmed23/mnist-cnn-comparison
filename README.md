# Handwritten Digit Recognition: Modern and Historical CNNs

This project compares François Chollet's published Keras Simple MNIST ConvNet with a compact modern Keras approximation of the convolutional architecture described by Le Cun et al. (1989).

## Project contents

- `report.pdf` — academic report with the experiment, measured results, limitations, process and resource reflection, references, and code appendix.
- `mnist_cnn_comparison.ipynb` — notebook with saved experiment outputs and visualizations.
- `project_cv.py` — reusable training, evaluation, and figure-generation code.
- `requirements.txt` — Python dependencies.
- `results/experiment_results.json` — run configuration, metrics, and histories.
- `results/layer_parameters.csv` — layer-level parameter counts.
- `figures/` — result plots.
- `SUBMISSION_CHECKLIST.md` — final manual checks before myCampus submission.

## Run

Python 3.12 is recommended. In Windows PowerShell, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python project_cv.py
```

Open the notebook in Jupyter and run all cells to inspect the saved outputs. To retrain both models, set `RUN_EXPERIMENT = True` in the setup cell or run `python project_cv.py`. The canonical MNIST 60,000/10,000 split is loaded through `tf.keras.datasets.mnist`; the initial download requires internet access.

## Reproducibility and scope

Both models use the same official MNIST split, 10% of the training data for validation, Adam, batch size 128, and 15 epochs. Seed 2026 is applied before model initialization. Training time is wall-clock `model.fit` time on the machine described in the report; it is environment-dependent. The test set is held out from model fitting and used for final evaluation.

The LeCun-inspired implementation approximates the paper's shared-weight 5x5 feature extraction and local averaging/subsampling using modern Keras layers. Dropout is used as the task brief's suggested proxy for the historical sparse H2-to-H3 connectivity. It is not an exact recreation of the original 1989 connection table or training procedure. The report describes one seeded run; the results are descriptive rather than evidence of statistical significance.
