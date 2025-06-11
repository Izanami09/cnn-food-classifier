# CNN Food Classifier

This project implements a Convolutional Neural Network (CNN) for classifying food images. It aims to accurately identify various food items from given images.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

* **Food Image Classification:** Classifies various food items using a deep learning model.
* **CNN Architecture:** Utilizes a Convolutional Neural Network built with PyTorch for robust image feature extraction.
* **Modular Design:** Organized into distinct modules for data handling, model definition, training, and utilities.

## Project Structure

The repository is structured as follows:

- cnn-food-classifier/
- ├── app/                  # Contains application logic (e.g., for deployment or inference)
- ├── models/               # Pre-trained models or model architectures
- ├── notebooks/            # Jupyter notebooks for experimentation, data exploration, or detailed analysis
- ├── utils/                # Utility functions (e.g., data loading, preprocessing)
- ├── pycache/          # Python bytecode cache
- ├── .DS_Store             # macOS specific hidden file
- ├── CustomDataSet.py      # Custom dataset handling script
- ├── model.py              # Defines the CNN model architecture
- ├── trainNN.py            # Script for training the neural network
- ├── README.md             # This README file
- └── requirements.txt      # List of project dependencies



## Technologies Used

* **Python**
* **PyTorch:** For building and training the deep learning model.
* **torchsummary:** For model summary visualization.
* **NumPy:** For numerical operations.
* **Pillow / OpenCV:** For image processing.
* **Matplotlib / Seaborn:** For data visualization.

## Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/izanami09/cnn-food-classifier.git](https://github.com/izanami09/cnn-food-classifier.git)
    cd cnn-food-classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```
    torch
    torchvision
    numpy
    pillow
    torchsummary
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the CNN model, you will typically run the `trainNN.py` script.


```bash
# Example command - **UPDATE THIS** with your actual training command
python trainNN.py --data_path /path/to/your/food_dataset/ --epochs 30 --batch_size 32
```

## Model Architecture Overview
The model (model0) is a Convolutional Neural Network designed for image classification. Its structure can be summarized using torchsummary:
```bash
summary(
    model= model0,
    input_size=(32,3,256,256), # Example input size: Batch_Size, Channels, Height, Width
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)
```



## Running the Inference Application (Streamlit)
To launch the interactive food classifier application:

Ensure you have trained a model and saved its weights (e.g., model.pth) in the models/ directory, or ensure your Streamlit app loads it correctly.
Navigate to the project root directory.
Run the Streamlit application:
```bash
streamlit run app/gui.py
```
This will open the application in your web browser, allowing you to upload food images for classification.

## Dataset
This project utilizes a custom food image dataset.

## The training process involves:

- Loss Function: nn.CrossEntropyLoss()
- Optimizer: torch.optim.Adam
- Learning Rate (lr): 0.0001
- Weight Decay: 1e-4
- Epochs: 30
- Device: Training is performed on the available device (CPU or GPU, determined by device variable).
- The core training loop is managed by the train function:

```python
results = train(
    model=model0,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    epochs=30,
    device=device
)
```

## Evaluation

## Result


## Contributing
- Contributions are welcome! If you'd like to contribute, please follow these steps:

* Fork the repository.
* Create a new branch (git checkout -b feature/AmazingFeature).
* Make your changes.
* Commit your changes (git commit -m 'Add some AmazingFeature'). 5. Push to the branch (git push origin feature/AmazingFeature).
* Open a Pull Request.


## License
This project is licensed under the MIT License

## 📬 Contact

* Rohan Kumar Shah
* 📧 rohan123.rs8@gmail.com
* 🐙 Izanami09

  - Made with ❤️ using Streamlit & Machine Learning




