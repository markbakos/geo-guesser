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

This project uses a deep learning approach to predict the geographic location of an image. By using convolutional neural networks (CNNs), it identifies patterns in visual data that
correlate with specific regions. The model, designed with a hierarchical structure predicts the coordinates and region (continent).
<br><br>
The dataset for training is collected using Mapillary API. The model's architecture combines pre-trained layers from EfficientNetV2S backbone with custom dense layers.
Outputs are weighted for a focus on regional and coordinate predictions.


## Features <a name = "features"></a>

**Data Handling:**

- Uses the Mapillary API to gather equal amount of diverse images per region for training
- Splits data for training, validation and testing.

**Deep Learning Model:**

- Employs EfficientNetV2S as the backbone, with additional layers for hierarchical predictions.
- Estimates latitude and longitude coordinate and predicts region. 
- Haversine Loss for coordinate accuracy and Categorical Crossentropy for regions.

**Training and Evaluation:**

- Tracks multiple metrics like regional accuracy and coordinate based location accuracy.
- Saves `best_location_model.keras` as the best validation coordinates accuracy from training.
- Saves `best_overall_model.keras` as the best overall model based on validation loss from training.

**Frontend UI:**
- Uses NextJS for the frontend, with TypeScript and TailwindCSS.

## Requirements <a name = "requirements"></a>

### Prerequisites
1. **Python 3.10 or higher**: Install from [python.org](https://www.python.org/downloads/).
2. **pip**: Python package manager (comes with Python installations).

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

2. **Set up enviromental variables**
- In the root folder (geo-guesser), in your .env file:
```
MAPILLARY_KEY=[Your API key]
```

3. **Install dependencies:**
```
pip install -r requirements.txt
```

4. **Set up the dataset**

- Collect images using `mapillary_collection.py`
- Recommended amount of images: 25-50k or more 


5. **Use the trained model**
- **From console**:
```
python -m predict path/to/saved/image
```

**Using the UI -- Either**:
- Use the deployed website: https://geo-guesser.onrender.com/
- Start the development server:

**Start the FastAPI server**
```
uvicorn server:app
```

## Contributing <a name = "contributing"></a>

Feel free to fork this repository, make changes, and submit a pull request.

## 📧 Contact <a name = "contact"></a>

For any inquiries, feel free to reach out:

Email: [markbakosss@gmail.com](mailto:markbakosss@gmail.com) <br>
GitHub: [markbakos](https://github.com/markbakos)
