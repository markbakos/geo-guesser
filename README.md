# Geo Guesser

<img src="https://wakatime.com/badge/user/7a2d5960-3199-4705-8543-83755e2b4d0c/project/c7d384e6-de45-4cc6-b013-bb6fa890bc67.svg" alt="Time spent in project" title="Time spent in project" />

## Table of Contents

+ [About](#about)
+ [Features](#features)
+ [Requirements](#requirements)
+ [Installation](#installation)
+ [Contributing](#contributing)
+ [Contact](#contact)

## About <a name = "about"></a>

This is a deep learning project that tries to predict where an image was taken. It uses a Convolutional Neural Network to analyze visual features and
distinguish between locations based on the unique architectural, environmental and infrastructural elements.

The project currently focuses on five major capitals (you can set custom cities and locations):
- **Budapest**
- **Ottawa**
- **Tokyo**
- **Cairo**
- **Canberra**

The model performs two key tasks:
1. **City Classification:** Assigns an image to one of the pre-defined city categories using a softmax-activated output layer.
2. **Coordinate Regression:** Estimates latitude and longitude values via a linear-activated output layer.

This project is divided into three main components:
- **Backend:** A FastAPI server that connects the model with the web interface.
- **Model:** The EfficientNetV2S-based neural network handling both classification and regression.
- **Frontend:** A web application built with Next.js, TypeScript and TailwindCSS to easily interact with the model.

<img src="https://github.com/markbakos/geo-guesser/blob/main/images/model.png?raw=true" alt="The model's architecture">

## Features <a name = "features"></a>

**Data Handling:**

- Uses the Mapillary API to collect street-level images and metadata using the Mapillary API.
- Speeds up data gathering through concurrent API requests.
- Preprocesses images (resizing to 224x224 pixels) for efficient training.

**Deep Learning Model:**

- Uses the pre-trained EfficientNetV2S network with custom upper layers.
- Employs fine-tuning where the EfficientNetV2S base is frozen, and only the custom layers are trained with the Adam optimizer.
- Uses GRAD-CAM to produce heatmaps on request that reveal image regions influencing the model's decisions.

<img src="https://github.com/markbakos/geo-guesser/blob/main/images/heatmap.png?raw=true" alt="Heatmap from the model">

**Training and Evaluation:**

- Achieves approximately 83% accuracy in city classification.
- Saves `best_location_model.keras` as the best validation coordinates accuracy from training.
- Saves `best_overall_model.keras` as the best overall model based on validation loss from training.

**Frontend UI:**
- Developed with Next.js, TypeScript and TailwindCSS.
- Provides an easily usable interface for users to interact with the model.
- Accessible online at [Location Guesser](https://locationguesser.vercel.app),

## Requirements <a name = "requirements"></a>

### Prerequisites
1. **Python 3.10 or higher**: Install from [python.org](https://www.python.org/downloads/).
2. **pip**: Python package manager (comes with Python installations).
3. **CUDA Toolkit 12.8** (optional)
4. **cuDNN 9.7.1** (optional)

### Python Dependencies

Install the required Python packages from `requirements.txt` found in the root folder.

```
pip install -r requirements.txt
```

## Installation <a name = "installation"></a>

1. **Clone the repository**
```
 https://github.com/markbakos/geo-guesser.git
 cd geo-guesser
```

2. **Set up environmental variables**
- In the root folder (geo-guesser), in your .env file:
```
MAPILLARY_KEY=[Your API key]
```

3. **Prepare the dataset**

- Set your desired locations to gather data from, or keep the original 5.
- Collect images using `mapillary_collection.py`

4. **Using the trained model**
- **From console**:
```
python -m predict path/to/saved/image --generate_heatmap
```

- **With the UI**:
  - Use the deployed website: <a target="_blank" href="https://locationguesser.vercel.app">https://locationguesser.vercel.app</a>
<br><br>
- **Start the FastAPI server:**
```
uvicorn server:app
```

## Contributing <a name = "contributing"></a>

Feel free to fork this repository, make changes, and submit a pull request.

## 📧 Contact <a name = "contact"></a>

For any inquiries, feel free to reach out:

Email: [markbakosss@gmail.com](mailto:markbakosss@gmail.com) <br>
GitHub: [markbakos](https://github.com/markbakos)
