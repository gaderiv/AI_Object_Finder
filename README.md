# AI_Object_Finder

This project aims to create a real-time theft detection system using deep learning models. It includes data preparation, model training, and inference scripts.

## Directory Structure

```plaintext
AI_Object_Finder/
│
├── src/
│   ├── main/
│   │   ├── python/
│   │   │   ├── data/
│   │   │   │   ├── prepare_data.py
│   │   │   │   └── datasets.py
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── i3d.py
│   │   │   │   ├── efficientdet.py
│   │   │   │   └── efficientdet_3d.py
│   │   │   │
│   │   │   ├── utils/
│   │   │   │   └── preprocessing.py
│   │   │   │
│   │   │   ├── train.py
│   │   │   └── infer.py
│   │   │
│   │   ├── resources/
│   │   │   └── DCSASS Dataset/
│   │   │   │   ├── Labels/
│   │   │   │   │   ├── Shoplifting.csv
│   │   │   │   │   └── Stealing.csv
│   │   │   │   └── Output/
│   │   │   │   │   ├── test.csv
│   │   │   │   │   ├── train.csv
│   │   │   │   │   ├── val.csv
│   │   │   │   │   └── second model/
│   │   │   │   │   │   ├── test.csv
│   │   │   │   │   │   ├── train.csv
│   │   │   │   │   │   └── val.csv
│   │
│   ├── test/
│   │   └── python/
│   │   │   └── test_sample.py
├── .gitignore
├── build.py
├── README.md
└── LICENSE

```
## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyBuilder
- Required Python packages (will be installed by PyBuilder)

### Installation

1. **Clone the Repository**:

    ```
    git clone https://github.com/yourusername/AI_Object_Finder.git
    cd AI_Object_Finder
    ```

2. **Install PyBuilder**:

    ```
    pip install pybuilder
    ```

### Running the Script

1. **Navigate to the Project Directory**:

    ```
    cd path_to_AI_Object_Finder
    ```

2. **Run PyBuilder**:

    ```
    pyb install_dependencies
    pyb clean build
    ```

### Preparing the Data

To prepare the video data for training:

    
    python src/main/python/data/prepare_data.py
    

### Training the Model

To train the model:

    
    python src/main/python/train.py
    

### Running Inference

To run inference on a video:

    
    python src/main/python/infer.py
    

### Running Unit Tests

To run the unit tests:

    
    python src/test/python/test_sample.py
    

## Project Structure

The project is structured as follows:

- `src/main/python/data`: Contains data preparation scripts and dataset definitions.
- `src/main/python/models`: Contains model definitions for I3D and EfficientDet.
- `src/main/python/utils`: Contains utility functions, e.g., preprocessing.
- `src/main/python/train.py`: Script for training the models.
- `src/main/python/infer.py`: Script for running inference on videos.
- `src/test/python`: Contains unit tests.
- `.gitignore`: Specifies which files and directories to ignore in the repository.
- `build.py`: PyBuilder build script.
- `README.md`: This file, providing an overview and instructions.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add some fooBar'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
