# MMDB
PCMMD: A Novel Dataset of Plasma Cells to Support the Diagnosis of Multiple Myeloma is a novel dataset of plasma cells.

This repository contains the database with annotations of plasma cells and non-plasma cells, as well as the diagnoses of 10 patients with multiple myeloma.

In addition to the database, a training was performed using the deep learning model Yolo version 8 for cell identification, and diagnosis based on the population of plasma cells.

## Project Structure

The following folder division was made for dataset organization and to conduct a benchmark with the goal of diagnosing 10 patients. Cross-validation with 5 folds was performed, and the model's variance against the dataset was checked.

- `data/`:
  - `detection/`: Data for object detection.
    - `patients/`: It contains the images, annotations of plasma cells and non-plasma cells, and the diagnoses of 10 patients.
    - `train/`:  It contains the images and annotations of plasma cells and non-plasma cells from various patients.
    - `mieloma_fold_{i}.yaml` Configuration file for each fold `i` used for training.
    - `split_{1}.txt`: File with the path of the images for each fold `i`.
    - `mieloma_final.yaml`: Final file used to train with all folds and test with the 10 patients.
  - `segmentation/`: Data for instance segmentation.
    - `plasma cells/`: It contains cropped images and masks of plasma cells.
    - `non-plasma cells/`: It contains cropped images and masks of non-plasma cells.
- `src`:
  - `dignosis.ipynb`: Jupyter notebook that performs the diagnosis of the 10 patients.
  - `plot.ipynb`: Jupyter notebook that visualizes detection and segmentation results for analysis and debugging.
  - `weights.pt`: Yolo v8 weights from the final training used for diagnosis.



### Benchmarking reproduction

#### Instalation

To setup and install Yolov8: [ultralytics](https://github.com/ultralytics/ultralytics)

---

#### Train:

To train model, use folds `mieloma_fold_{i}.yaml` or final file `mieloma_final.yaml` in Yolov8.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="mieloma_final.yaml.yaml", epochs=300)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

```



#### Test

Use `MMDB/src/diagnosis.ipynb` for test.
