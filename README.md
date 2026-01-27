<div align="center">

  <a name="readme-top"></a>
  # Depression Detection Using Tweets

  [![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
  ![Status](https://img.shields.io/badge/Status-Completed-success)
  [![Technology](https://img.shields.io/badge/Technology-Python%20%7C%20Machine%20Learning-blueviolet)](https://github.com/Amey-Thakur/DEPRESSION_DETECTION_USING_TWEETS)
  [![Developed by Amey Thakur and Mega Satish](https://img.shields.io/badge/Developed%20by-Amey%20Thakur%20%26%20Mega%20Satish-blue.svg)](https://github.com/Amey-Thakur/DEPRESSION_DETECTION_USING_TWEETS)

  A modern **Python** + **Flask** application designed to analyze tweet sentiment and predict depressive characteristics using a finalized **SVM** model and **spaCy** NLP pipeline.

  **[Source Code](Source%20Code/)** &nbsp;·&nbsp; **[Technical Specification](docs/SPECIFICATION.md)** &nbsp;·&nbsp; **[Live Demo](https://huggingface.co/spaces/ameythakur/Depression-Detection-Using-Tweets)**

</div>

---

<div align="center">

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

</div>

---

<!-- AUTHORS -->
<div align="center">

  <a name="authors"></a>
  ## Authors

| <a href="https://github.com/Amey-Thakur"><img src="https://github.com/Amey-Thakur.png" width="150" height="150" alt="Amey Thakur"></a><br>[**Amey Thakur**](https://github.com/Amey-Thakur)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-green.svg)](https://orcid.org/0000-0001-5644-1575) | <a href="https://github.com/msatmod"><img src="Mega/Mega.png" width="150" height="150" alt="Mega Satish"></a><br>[**Mega Satish**](https://github.com/msatmod)<br><br>[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--1844--9557-green.svg)](https://orcid.org/0000-0002-1844-9557) |
| :---: | :---: |

</div>

> [!IMPORTANT]
> ### 🤝🏻 Special Acknowledgement
> *Special thanks to **[Mega Satish](https://github.com/msatmod)** for her meaningful contributions, guidance, and support that helped shape this work.*

---

<!-- OVERVIEW -->
<a name="overview"></a>
## Overview

**Depression Detection Using Tweets** is a specialized Machine Learning framework designed to translate complex linguistic patterns into empirical psychological insights. This repository prioritizes **high-dimensional feature extraction** and **probabilistic classification** to provide a robust baseline for sentiment analysis within the context of mental health monitoring.

*   **Linguistic Determinism**: The system utilizes deep NLP preprocessing, including lemmatization and entity normalization, to ensure that the semantic core of a tweet is preserved regardless of slang or stylistic variation.
*   **Vector-Space Inference**: By leveraging **Support Vector Machines (SVM)** and **TF-IDF vectorization**, the model maps textual input into a multi-dimensional hyperplane, enabling precise binary classification of depressive sentiment.
*   **Architectural Efficiency**: The backend is architected for low-latency serving via Flask, ensuring that model inference and result rendering occur in sub-second cycles, critical for interactive user feedback.

> [!TIP]
> **NLP Pipeline Optimization**
>
> To maximize classification reliability, the engine employs a **multi-stage linguistic filter**. **Stop-word suppression** and **morphological analysis** strip away structural noise, while the **en_core_web_lg** transformer model contextualizes surviving tokens. This ensures the classifier’s weights are strictly coupled with affective indicators, minimizing the false-positive skew common in generalized sentiment analysis models.

---

<!-- FEATURES -->
<a name="features"></a>
## Features

| Feature | Description |
|---------|-------------|
| **Core SVM Model** | **High-Dimensional Classification** engine optimized for binary depressive sentiment prediction. |
| **NLP Pipeline** | Deep linguistic feature extraction powered by the **spaCy transformer model** (`en_core_web_lg`). |
| **Prediction Hub** | **Real-Time Inference Interface** built with Flask for sub-second classification feedback. |
| **Security Suite** | Integrated **Browser-Side Integrity** protocols including anti-right-click and anti-select systems. |
| **Cinematic Surprise** | **Immersive Branding Overlay** featuring animated Twitter iconography and synchronized audio. |

> [!NOTE]
> ### Technical Polish: The Linguistic Singularity
> We have engineered a **Probabilistic Sentiment Manager** that calibrates model weights across thousands of TF-IDF vectors to simulate human-like linguistic intuition. The visual language focuses on a "Neural Slate" aesthetic, ensuring maximum cognitive focus on the diagnostic outputs without procedural distraction.

### Tech Stack
- **Languages**: Python 3.9+
- **Logic**: **SVM Classifier** (Scikit-Learn Inference Engine)
- **Linguistic Data**: **spaCy NLP** (Transformer-based word embeddings)
- **Web App**: **Flask Framework** (Micro-service architecture for model serving)
- **UI System**: Premium Modern Aesthetics (Custom CSS / Play Typography)
- **Deployment**: Standard Python Environment (PIP-managed dependencies)

---

<!-- PROJECT STRUCTURE -->
<a name="project-structure"></a>
## Project Structure

```python
DEPRESSION-DETECTION-USING-TWEETS/
│
├── docs/                            # Technical Documentation
│   └── SPECIFICATION.md             # Architecture & Design Specification
│
├── Mega/                            # Archival Attribution Assets
│   ├── Filly.jpg                    # Companion (Filly)
│   └── Mega.png                     # Author Profile Image (Mega Satish)
│
├── screenshots/                     # Project Visualization Gallery
│   ├── 01_landing_page.png          # System Hub Initial State
│   ├── 02_footer_details.png        # Brand and Metadata Footer
│   ├── 03_surprise_cinematic.png    # Interactive Animated Sequence
│   ├── 04_predict_interface.png     # Sentiment Analysis Entry Point
│   ├── 05_analysis_output.png       # Model Inference result
│   └── 06_result_prediction.png     # Final Sentiment Output
│
├── Source Code/                     # Primary Application Layer
│   ├── assets/                      # Serialized Models & Linguistic Data
│   ├── core/                        # ML Pipeline (Clean, Train, Predict)
│   ├── static/                      # Styling, Audio, & Security Scripts
│   ├── templates/                   # HTML Templates (Index, Result, 404)
│   └── app.py                       # Flask Application (Entry Point)
│
├── .gitattributes                   # Git configuration
├── .gitignore                       # Repository Filters
├── CITATION.cff                     # Scholarly Citation Metadata
├── codemeta.json                    # Machine-Readable Project Metadata
├── LICENSE                          # MIT License Terms
├── README.md                        # Comprehensive Scholarly Entrance
└── SECURITY.md                      # Security Policy & Protocol
```

---

<!-- RESULTS -->
<a name="results"></a>
## Results

<div align="center">
  <b>Main Landing: System Hub Initialization</b>
  <br>
  <i>Minimalist interface for rapid tweet sentiment analysis.</i>
  <br><br>
  <img src="screenshots/01_landing_page.png" alt="Landing Page" width="90%">
  <br><br><br>

  <b>Metadata Synthesis: Branding and Footer Detail</b>
  <br>
  <i>Scholarly attribution and project status integration.</i>
  <br><br>
  <img src="screenshots/02_footer_details.png" alt="Footer Details" width="90%">
  <br><br><br>

  <b>Interactivity: Animated Twitter Sequence</b>
  <br>
  <i>Immersive audiovisual overlay triggered by core branding elements.</i>
  <br><br>
  <img src="screenshots/03_surprise_cinematic.png" alt="Cinematic Surprise" width="90%">
  <br><br><br>

  <b>Sentiment Entry: Real-time Analysis Interface</b>
  <br>
  <i>Direct manipulation environment for high-latency textual input.</i>
  <br><br>
  <img src="screenshots/04_predict_interface.png" alt="Predict Interface" width="90%">
  <br><br><br>

  <b>Model Inference: Feature Extraction Output</b>
  <br>
  <i>Deep linguistic analysis and probabilistic score generation.</i>
  <br><br>
  <img src="screenshots/05_analysis_output.png" alt="Analysis Output" width="90%">
  <br><br><br>

  <b>Statistical Output: Final Sentiment Classification</b>
  <br>
  <i>Categorized classification results with immediate visual feedback.</i>
  <br><br>
  <img src="screenshots/06_result_prediction.png" alt="Result Prediction" width="90%">
</div>

---

<!-- QUICK START -->
<a name="quick-start"></a>
## Quick Start

### 1. Prerequisites
- **Python 3.11+**: Required for runtime execution. [Download Python](https://www.python.org/downloads/)
- **Git**: For version control and cloning. [Download Git](https://git-scm.com/downloads)

> [!WARNING]
> **Data Acquisition & Memory Constraints**
>
> The linguistic pipeline relies on the **en_core_web_lg** transformer model, which requires an initial download of approximately **800MB**. Ensure a stable network connection during setup. Additionally, loading this model into memory requires at least **2GB of available RAM** to prevent swapping and ensure low-latency inference.

### 2. Installation & Setup

#### Step 1: Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/Amey-Thakur/DEPRESSION-DETECTION-USING-TWEETS.git
cd DEPRESSION-DETECTION-USING-TWEETS
```

#### Step 2: Configure Virtual Environment
Prepare an isolated environment to manage dependencies:

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux (Terminal):**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Core Dependencies
Ensure your environment is active, then install the required libraries:
```bash
pip install -r "Source Code/requirements.txt"
```

#### Step 4: Linguistic Model Acquisition
Download the large-scale linguistic model required for analysis (approx. 800MB):
```bash
python -m spacy download en_core_web_lg
```

### 3. Execution
Launch the sentiment analysis dashboard:

```bash
python "Source Code/app.py"
```

---

<!-- USAGE GUIDELINES -->
<a name="usage-guidelines"></a>
## Usage Guidelines

This repository is openly shared to support learning and knowledge exchange across the academic community.

**For Students**  
Use this project as reference material for understanding **Support Vector Machines (SVM)**, **spaCy NLP pipelines**, and **sentiment analysis within the context of mental health monitoring**. The source code is available for study to facilitate self-paced learning and exploration of **high-dimensional feature extraction and model serving via Flask**.

**For Educators**  
This project may serve as a practical lab example or supplementary teaching resource for **Data Science**, **Natural Language Processing**, and **Machine Learning** courses. Attribution is appreciated when utilizing content.

**For Researchers**  
The documentation and architectural approach may provide insights into **academic project structuring**, **psychological linguistic modeling**, and **algorithmic deployment**.

---

<!-- LICENSE -->
<a name="license"></a>
## License

This repository and all its creative and technical assets are made available under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms.

> [!NOTE]
> **Summary**: You are free to share and adapt this content for any purpose, even commercially, as long as you provide appropriate attribution to the original authors.

Copyright © 2022 Amey Thakur & Mega Satish

---

<!-- ABOUT -->
<a name="about-this-repository"></a>
## About This Repository

**Created & Maintained by**: [Amey Thakur](https://github.com/Amey-Thakur) & [Mega Satish](https://github.com/msatmod)

This project features **Depression Detection**, a high-performance sentiment analysis system. It represents a personal exploration into **Python**-based machine learning and interactive web-service architecture.

**Connect:** [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [LinkedIn](https://www.linkedin.com/in/amey-thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

### Acknowledgments

Grateful acknowledgment to [**Mega Satish**](https://github.com/msatmod) for her exceptional collaboration and scholarly partnership during the development of this machine learning project. Her constant support, technical clarity, and dedication to software quality were instrumental in achieving the system's functional objectives. Learning alongside her was a transformative experience; her thoughtful approach to problem-solving and steady encouragement turned complex requirements into meaningful learning moments. This work reflects the growth and insights gained from our side-by-side academic journey. Thank you, Mega, for everything you shared and taught along the way.

Special thanks to the **mentors and peers** whose encouragement, discussions, and support contributed meaningfully to this learning experience.

---

<div align="center">

  [↑ Back to Top](#readme-top)

  [Authors](#authors) &nbsp;·&nbsp; [Overview](#overview) &nbsp;·&nbsp; [Features](#features) &nbsp;·&nbsp; [Structure](#project-structure) &nbsp;·&nbsp; [Results](#results) &nbsp;·&nbsp; [Quick Start](#quick-start) &nbsp;·&nbsp; [Usage Guidelines](#usage-guidelines) &nbsp;·&nbsp; [License](#license) &nbsp;·&nbsp; [About](#about-this-repository) &nbsp;·&nbsp; [Acknowledgments](#acknowledgments)

  <br>

  🧠 **[DEPRESSION-DETECTION](https://huggingface.co/spaces/ameythakur/Depression-Detection-Using-Tweets)**

  ---

  ### 🎓 [Computer Engineering Repository](https://github.com/Amey-Thakur/COMPUTER-ENGINEERING)

  **Computer Engineering (B.E.) - University of Mumbai**

  *Semester-wise curriculum, laboratories, projects, and academic notes.*

</div>