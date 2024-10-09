# GA-Feature-Selector-Lung-Cancer

This repository contains the implementation of a **Genetic Algorithm-based Deep Feature Selector** designed to optimize feature selection for lung cancer detection using histopathological images. This approach combines the power of genetic algorithms with deep learning to enhance classification accuracy while minimizing the feature set.

## Features
- Genetic Algorithm (GA) for selecting optimal features from deep learning models.
- Deep feature extraction using convolutional neural networks (CNNs).
- Application for detecting cancer in lung histopathological images.
- High-performance classification with a reduced feature set, improving computational efficiency.

## Repository Structure
- `GA_FeatureSelector_LungCancer.ipynb`: Jupyter notebook that contains the entire pipeline, including data preprocessing, feature extraction, genetic algorithm for feature selection, and classification.
- `data/`: Directory for storing lung histopathological image data (not included in the repo).
- `models/`: Directory to store pre-trained models or save newly trained models.
- `results/`: Directory to save performance results, including feature selection outcomes and classification metrics.

## Requirements
The project requires the following Python libraries:
- Python 3.x
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- Numpy
- Scikit-learn
- Matplotlib

You can install the required dependencies by running:
```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GA-Feature-Selector-Lung-Cancer.git
   ```

2. Navigate to the project directory:
   ```bash
   cd GA-Feature-Selector-Lung-Cancer
   ```

3. Run Jupyter Notebook:
   ```bash
   jupyter-notebook
   ```

4. In the Jupyter interface, open the file `GA_FeatureSelector_LungCancer.ipynb`.

5. Follow the notebook cells to:
   - Preprocess the lung histopathological image data.
   - Extract features using deep learning models.
   - Perform feature selection using a Genetic Algorithm.
   - Train and evaluate the classifier with the selected features.

## Data

- You need to provide the lung histopathological image data. Place the images inside the `data/` directory.
- Make sure the data is in the correct format, or modify the notebook’s data loading section accordingly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
