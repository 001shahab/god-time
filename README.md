# Divine Time Mapping

A beautiful, elegant web application that visualizes the mathematical relationship between Biblical creation timeline (6 days) and scientific cosmic timeline (13.8 billion years) using concepts from Special and General Relativity.

![3S Holding OÜ](static/img/logo.png)

## Overview

This application provides an interactive exploration of how the six days of creation in Biblical narrative can be mathematically mapped to the 13.8 billion years of cosmic history through relativistic time dilation and exponential mapping models.

### Key Concepts

- **Divine Days**: 6 days of creation
- **Scientific Age**: 13.8 billion years
- **Lorentz Factor (γ)**: ~8.4 × 10¹¹
- **Base Time Constant (T₀)**: ~219 million years

### The Mapping Equation

```
t_S = T₀ × (2^(t_G) − 1)
```

Where:
- `t_S` = Scientific time (billion years)
- `t_G` = Divine time (days)
- `T₀` = Base constant (~0.219 Gyr)

Each creation day doubles in cosmic duration, creating an exponential relationship where:
- Day 1: 0.22 billion years
- Day 2: 0.44 billion years
- Day 3: 0.88 billion years
- Day 4: 1.75 billion years
- Day 5: 3.50 billion years
- Day 6: 7.01 billion years (more than half of cosmic history!)

## Features

- **Interactive Time Converter**: Convert between divine days and scientific years
- **Mapping Visualization**: See the exponential curve relating both timescales
- **Timeline Correspondence**: Biblical events mapped to scientific cosmology
- **Duration Analysis**: Understand how each day doubles the previous
- **Relativity Framework**: Explore the Lorentz factor and time dilation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/god-time.git
cd god-time
```

2. Create a virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5002
```

## Project Structure

```
god-time/
├── app.py                    # Flask application with API endpoints
├── divine_time_mapping.py    # Original simulation with matplotlib
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── templates/
│   └── index.html            # Main UI template
└── static/
    └── img/
        └── logo.png          # 3S Holding OÜ logo
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application page |
| `/api/constants` | GET | Fundamental parameters and constants |
| `/api/mapping` | GET | Complete time mapping data |
| `/api/convert` | POST | Convert between divine and scientific time |
| `/api/curve` | GET | Data points for mapping curve |
| `/api/relativity` | GET | Relativity visualization data |
| `/api/timeline` | GET | Timeline correspondence data |

## Design Philosophy

The UI is designed following Steve Jobs' design principles:
- **Simplicity**: Clean, uncluttered interface
- **Elegance**: Beautiful typography and subtle animations
- **Intuitiveness**: Self-explanatory interactions
- **Focus**: Content takes center stage

## Technologies

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Visualization**: Chart.js
- **Typography**: SF Pro Display, Cormorant Garamond

## License

© 3S Holding OÜ

## Author

Design and Developed by **Prof. Shahab Anbarjafari**
