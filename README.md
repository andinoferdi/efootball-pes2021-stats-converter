# eFootball to PES 2021 Stats Converter

Konverter statistik pemain dari eFootball 2025 ke PES 2021 menggunakan Machine Learning (ML) dengan ensemble model (ExtraTrees + Ridge Regression).

## Fitur

- Konversi statistik eFootball ke PES 2021 dengan akurasi tinggi
- Machine Learning ensemble model (ExtraTrees + Ridge)
- Web interface dengan Flask
- Input mode: JSON atau Manual (form input)
- Handling khusus untuk kiper (stat non-GK lebih realistis)
- Auto-clamp stat >= 99 di eFootball menjadi 99 di PES 2021
- Overall rating calculator untuk PES 2021
- Dataset builder dari PESMaster (scraper)

## Teknologi

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (ExtraTreesRegressor, Ridge)
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Data Processing**: NumPy, BeautifulSoup4

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/andinoferdi/efootball-pes2021-stats-converter.git
cd efootball-pes2021-stats-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Cara Menggunakan

### Web Interface

1. Jalankan server:
```bash
python app.py
```

2. Buka browser:
```
http://localhost:5000
```

3. Input stat eFootball:
   - **Tab JSON**: Paste JSON stat eFootball
   - **Tab Manual**: Isi form input untuk setiap stat
4. Pilih posisi pemain
5. (Opsional) Masukkan Overall Rating eFootball
6. Klik "Konversi"

### CLI Mode

#### Scrape Data dari PESMaster

```bash
python app.py --scrape --urls urls.txt --out dataset.jsonl
```

File `urls.txt` berisi URL PESMaster eFootball player pages (satu per baris).

#### Train Model

```bash
python app.py --train --data dataset.jsonl --model model.joblib
```

Minimal 8 training samples diperlukan untuk training yang stabil.

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
  "markdown_table": "| Atribut PES 2021 | ..."
}
```

## Fitur Khusus

### Handling Kiper

Stat non-GK untuk kiper menggunakan formula khusus yang lebih konservatif agar lebih realistis. Contoh:
- Ball Control 60 di eFootball → ~68 di PES 2021 (bukan ~83)

### Auto Clamp

Stat eFootball >= 99 otomatis di-clamp menjadi 99 di PES 2021 (maksimal PES 2021).

### Overall Rating

Overall rating PES 2021 dihitung berdasarkan posisi pemain dengan bobot berbeda untuk setiap statistik.

## Struktur Proyek

```
.
├── app.py                 # Main application (Flask + ML)
├── requirements.txt       # Python dependencies
├── README.md             # Dokumentasi
├── templates/
│   └── index.html        # Web interface
└── static/
    ├── css/
    │   └── custom.css    # Custom styling
    └── js/
        └── main.js       # Frontend JavaScript
```

## Model Machine Learning

- **Ensemble**: ExtraTreesRegressor (60%) + Ridge Regression (40%)
- **Features**: Raw stats, baseline guess, position encoding, interactions
- **Training**: Minimal 8 samples, optimal 20+ samples

## Lisensi

MIT License

## Kontribusi

Kontribusi sangat diterima. Silakan buat issue atau pull request.

## Catatan

- Model ML memerlukan training data untuk akurasi optimal
- Tanpa model, sistem menggunakan baseline formula
- Scraper PESMaster memerlukan koneksi internet

