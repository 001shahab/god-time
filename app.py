"""
Divine Time Mapping - Flask Application
A beautiful, elegant interface for exploring the relationship between
Biblical creation timeline and scientific cosmic timeline.

© 3S Holding OÜ - Design and Developed by Prof. Shahab Anbarjafari
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

c = 299792458  # Speed of light (m/s)
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 3.156e7

# Biblical timeline
DIVINE_DAYS = 6
DIVINE_TIME_SECONDS = DIVINE_DAYS * SECONDS_PER_DAY

# Scientific timeline
COSMIC_AGE_YEARS = 13.8e9  # 13.8 billion years
COSMIC_AGE_SECONDS = COSMIC_AGE_YEARS * SECONDS_PER_YEAR

# Derived quantities
GAMMA = COSMIC_AGE_SECONDS / DIVINE_TIME_SECONDS
BETA = np.sqrt(1 - 1/GAMMA**2)
T0_YEARS = COSMIC_AGE_YEARS / (2**6 - 1)

# Biblical events for each day
BIBLICAL_EVENTS = [
    {"day": 1, "title": "Light from Darkness", "description": "And God said, 'Let there be light'"},
    {"day": 2, "title": "Firmament (Expanse)", "description": "God made the expanse and separated the waters"},
    {"day": 3, "title": "Land & Vegetation", "description": "Let the dry land appear with vegetation"},
    {"day": 4, "title": "Sun, Moon & Stars", "description": "Lights in the expanse of the heavens"},
    {"day": 5, "title": "Sea Creatures & Birds", "description": "Let the waters swarm with living creatures"},
    {"day": 6, "title": "Land Animals & Humans", "description": "Let us make man in our image"}
]

# Scientific correlations
SCIENTIFIC_EVENTS = [
    {"day": 1, "title": "Big Bang & First Light", "description": "~380,000 years: Recombination, CMB release"},
    {"day": 2, "title": "Universe Expansion", "description": "Cosmic structure begins forming"},
    {"day": 3, "title": "Galaxy Formation", "description": "Heavy elements synthesized in stars"},
    {"day": 4, "title": "Solar System Formation", "description": "~4.6 Gya: Sun and planets form"},
    {"day": 5, "title": "Early Life Evolution", "description": "~3.8-0.5 Gya: Cambrian explosion"},
    {"day": 6, "title": "Complex Life & Humans", "description": "~300 Mya → 300 Kya: Evolution of complex life"}
]

# =============================================================================
# MAPPING FUNCTIONS
# =============================================================================

def divine_to_scientific(t_divine_days):
    """Convert divine time (days) to scientific time (billion years)."""
    return T0_YEARS * (2**t_divine_days - 1) / 1e9


def scientific_to_divine(t_scientific_gyr):
    """Inverse mapping: scientific time to divine time."""
    t_scientific_years = t_scientific_gyr * 1e9
    return np.log2(t_scientific_years / T0_YEARS + 1)


def day_duration_cosmic(day_number):
    """Calculate the cosmic duration of a specific creation day."""
    return T0_YEARS * 2**(day_number - 1) / 1e9


def lorentz_factor(v):
    """Calculate Lorentz factor for velocity v (as fraction of c)."""
    return 1 / np.sqrt(1 - v**2)


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/constants')
def get_constants():
    """Return fundamental constants and parameters."""
    return jsonify({
        'divine_days': DIVINE_DAYS,
        'cosmic_age_gyr': COSMIC_AGE_YEARS / 1e9,
        'gamma': float(GAMMA),
        'beta': float(BETA),
        'velocity_diff_from_c': float(1 - BETA),
        't0_gyr': T0_YEARS / 1e9
    })


@app.route('/api/mapping')
def get_mapping():
    """Return the complete time mapping data."""
    days = list(range(1, 7))
    mapping_data = []
    
    for day in days:
        start_gyr = divine_to_scientific(day - 1)
        end_gyr = divine_to_scientific(day)
        duration_gyr = end_gyr - start_gyr
        
        mapping_data.append({
            'day': day,
            'start_gyr': round(float(start_gyr), 4),
            'end_gyr': round(float(end_gyr), 4),
            'duration_gyr': round(float(duration_gyr), 4),
            'biblical_event': BIBLICAL_EVENTS[day - 1],
            'scientific_event': SCIENTIFIC_EVENTS[day - 1]
        })
    
    return jsonify({
        'mapping': mapping_data,
        'total_divine_days': DIVINE_DAYS,
        'total_cosmic_gyr': round(divine_to_scientific(6), 4)
    })


@app.route('/api/convert', methods=['POST'])
def convert_time():
    """Convert between divine and scientific time."""
    data = request.get_json()
    
    if 'divine_days' in data:
        divine_days = float(data['divine_days'])
        scientific_gyr = divine_to_scientific(divine_days)
        return jsonify({
            'divine_days': divine_days,
            'scientific_gyr': round(float(scientific_gyr), 6)
        })
    
    elif 'scientific_gyr' in data:
        scientific_gyr = float(data['scientific_gyr'])
        divine_days = scientific_to_divine(scientific_gyr)
        return jsonify({
            'scientific_gyr': scientific_gyr,
            'divine_days': round(float(divine_days), 6)
        })
    
    return jsonify({'error': 'Invalid input'}), 400


@app.route('/api/curve')
def get_curve():
    """Return points for the mapping curve visualization."""
    points = []
    t_divine = np.linspace(0, 6, 200)
    t_scientific = [divine_to_scientific(t) for t in t_divine]
    
    for i in range(len(t_divine)):
        points.append({
            'divine': round(float(t_divine[i]), 4),
            'scientific': round(float(t_scientific[i]), 4)
        })
    
    # Day boundaries
    boundaries = []
    for day in range(0, 7):
        boundaries.append({
            'day': day,
            'scientific_gyr': round(float(divine_to_scientific(day)), 4)
        })
    
    return jsonify({
        'curve_points': points,
        'boundaries': boundaries
    })


@app.route('/api/relativity')
def get_relativity_data():
    """Return data for relativity visualizations."""
    # Lorentz factor vs velocity
    beta_values = np.linspace(0, 0.9999, 100)
    gamma_values = [lorentz_factor(b) for b in beta_values]
    
    lorentz_curve = [
        {'beta': round(float(b), 6), 'gamma': round(float(g), 2)}
        for b, g in zip(beta_values, gamma_values)
    ]
    
    # Day durations
    day_durations = [
        {'day': d, 'duration_gyr': round(day_duration_cosmic(d), 4)}
        for d in range(1, 7)
    ]
    
    # Duration ratios (each day is 2x the previous)
    ratios = [
        {'transition': f'Day {d}→{d+1}', 'ratio': round(day_duration_cosmic(d+1) / day_duration_cosmic(d), 6)}
        for d in range(1, 6)
    ]
    
    return jsonify({
        'lorentz_curve': lorentz_curve,
        'day_durations': day_durations,
        'duration_ratios': ratios,
        'divine_frame_gamma': float(GAMMA),
        'divine_frame_beta': float(BETA)
    })


@app.route('/api/timeline')
def get_timeline():
    """Return data for the dual timeline visualization."""
    timeline_data = []
    total_cosmic = 13.8
    
    for day in range(1, 7):
        start_gyr = divine_to_scientific(day - 1)
        end_gyr = divine_to_scientific(day)
        duration_gyr = end_gyr - start_gyr
        
        timeline_data.append({
            'day': day,
            'divine_start': day - 1,
            'divine_end': day,
            'cosmic_start_gyr': round(float(start_gyr), 4),
            'cosmic_end_gyr': round(float(end_gyr), 4),
            'cosmic_duration_gyr': round(float(duration_gyr), 4),
            'cosmic_percentage': round(float(duration_gyr / total_cosmic * 100), 2),
            'biblical': BIBLICAL_EVENTS[day - 1],
            'scientific': SCIENTIFIC_EVENTS[day - 1]
        })
    
    return jsonify(timeline_data)


if __name__ == '__main__':
    app.run(debug=True, port=5002)

