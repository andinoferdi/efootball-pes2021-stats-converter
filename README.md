# eFootball to PES 2021 Stats Converter

Convert player statistics from eFootball 2025 to PES 2021 using Machine Learning (ML) with ensemble model (ExtraTrees + Ridge Regression).

## Features

- High accuracy eFootball to PES 2021 statistics conversion
- Machine Learning ensemble model (ExtraTrees + Ridge)
- Modern web interface with Flask
- Dark mode support with theme toggle
- Input modes: JSON or Manual (form input)
- Special handling for goalkeepers (more realistic non-GK stats)
- Auto-clamp stats >= 99 in eFootball to 99 in PES 2021
- Overall rating calculator for PES 2021
- Dataset builder from PESMaster (scraper)

## Technologies

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (ExtraTreesRegressor, Ridge)
- **Frontend**: HTML, Tailwind CSS, JavaScript (modular architecture)
- **Styling**: Modular CSS (base, components, animations)
- **Data Processing**: NumPy, BeautifulSoup4

## Installation

1. Clone repository:
```bash
git clone https://github.com/andinoferdi/efootball-pes2021-stats-converter.git
cd efootball-pes2021-stats-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Run server:
```bash
python app.py
```

2. Open browser:
```
http://localhost:5000
```

3. Input eFootball stats:
   - **JSON Tab**: Paste JSON eFootball stats
   - **Manual Tab**: Fill form inputs for each stat
4. Select player position
5. (Optional) Enter eFootball Overall Rating
6. Click "Convert Stats"

### Dark Mode

The interface includes dark mode support by default. Use the theme toggle button in the top-right corner to switch between light and dark themes. Your preference is saved in localStorage.

### CLI Mode

#### Scrape Data from PESMaster

```bash
python app.py --scrape --urls urls.txt --out dataset.jsonl
```

File `urls.txt` contains PESMaster eFootball player page URLs (one per line).

#### Train Model

```bash
python app.py --train --data dataset.jsonl --model model.joblib
```

Minimum 8 training samples required for stable training.

## Format Input JSON

```json
{
  "offensive_awareness": 96,
  "ball_control": 84,
  "dribbling": 89,
  "tight_possession": 90,
  "low_pass": 95,
  "lofted_pass": 99,
  "finishing": 85,
  "heading": 62,
  "place_kicking": 94,
  "curl": 100,
  "speed": 89,
  "acceleration": 86,
  "kicking_power": 95,
  "jump": 59,
  "physical_contact": 67,
  "balance": 82,
  "stamina": 94,
  "defensive_awareness": 60,
  "defensive_engagement": 60,
  "tackling": 56,
  "aggression": 64,
  "goalkeeping": 41,
  "gk_catching": 41,
  "gk_parrying": 41,
  "gk_reflexes": 41,
  "gk_reach": 41,
  "weak_foot_usage": 0,
  "weak_foot_acc": 1,
  "form": 2,
  "injury_resistance": 1
}
```

## API Endpoint

### POST /api/convert

Request body:
```json
{
  "ef_stats": {
    "offensive_awareness": 96,
    "ball_control": 84,
    ...
  },
  "position": "RMF",
  "ef_overall": 100
}
```

Response:
```json
{
  "pes_stats": {
    "Offensive Awareness": 92,
    "Ball Control": 93,
    ...
  },
  "pes_overall": 95,
  "markdown_table": "| PES 2021 Attribute | Converted Value | eFootball Value |"
}
```

## Special Features

### Goalkeeper Handling

Non-GK stats for goalkeepers use a special formula that is more conservative for realism. Example:
- Ball Control 60 in eFootball → ~68 in PES 2021 (not ~83)

### Auto Clamp

eFootball stats >= 99 are automatically clamped to 99 in PES 2021 (PES 2021 maximum).

### Overall Rating

PES 2021 overall rating is calculated based on player position with different weights for each statistic.

## Project Structure

```
.
├── app.py                 # Main application (Flask + ML)
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Web interface
└── static/
    ├── css/
    │   ├── base.css      # Base styles and CSS variables
    │   ├── components.css # Component styles
    │   └── animations.css # Animation styles
    └── js/
        ├── theme.js      # Theme management (dark mode)
        ├── utils.js      # Utility functions
        ├── tabs.js       # Tab functionality
        ├── accordion.js  # Accordion functionality
        ├── form.js       # Form handling
        └── effects.js     # Visual effects
```

## Machine Learning Model

- **Ensemble**: ExtraTreesRegressor (60%) + Ridge Regression (40%)
- **Features**: Raw stats, baseline guess, position encoding, interactions
- **Training**: Minimum 8 samples, optimal 20+ samples

## License

MIT License

## Contributing

Contributions are welcome. Please create an issue or pull request.

## Notes

- ML model requires training data for optimal accuracy
- Without model, system uses baseline formula
- PESMaster scraper requires internet connection
- Dark mode is enabled by default

