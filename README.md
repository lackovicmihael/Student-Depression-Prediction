# Student Depression Prediction

Projekt za kolegij **Računarstvo usluga i analiza podataka**. Cilj projekta je predikcija klase `Depression` nad tabličnim skupom podataka o studentima, uz treniranje i usporedbu više modela strojnog učenja te izlaganje najboljeg modela preko Azure ML Managed Online Endpointa.

Model služi isključivo za demonstraciju strojnog učenja i nije medicinski dijagnostički alat.

## Sažetak rješenja

Korišteni tijek rada:

```text
student_depression_dataset.csv
→ čišćenje i predobrada podataka
→ usporedba klasifikacijskih modela
→ odabir najboljeg modela prema F1 mjeri
→ Azure ML model registry
→ Azure ML Managed Online Endpoint
→ Streamlit klijentska aplikacija
```

Azure resursi korišteni u završnoj verziji:

- Workspace: `mlw-student-depression-prediction`
- Registered model: `student-depression-model-04`
- Endpoint: `student-depression-endpoint`
- Environment: `student-depression-sklearn-1-5`

## Struktura projekta

```text
azure/                  Azure environment i score.py za endpoint
data/                   CSV skup podataka
notebooks/              EDA notebook
outputs/eda/            slike deskriptivne analize
outputs/local_model/    spremljeni model, metrike i evaluacijski grafovi
scripts/                lokalna provjera score.py skripte
src/                    kod za predobradu, treniranje i evaluaciju
streamlit_app/          Streamlit REST klijent
```

## Lokalno pokretanje

```bash
python -m venv .venv
source .venv\Scripts\activate
pip install -r requirements-local.txt
```

Treniranje modela:

```bash
python -m src.train --data-path data/student_depression_dataset.csv --output-dir outputs/local_model --search-level quick --primary-metric f1
```

Lokalna provjera `score.py` skripte:

```bash
python scripts/test_score_local.py
```

Streamlit klijent:

```bash
streamlit run streamlit_app/app.py
```

U aplikaciju se unosi Azure API key iz Azure ML **Consume** taba. Ključ se ne sprema u repozitorij.

## Rezultati

Najbolji model u spremljenom eksperimentu je `XGBoost`. Ostvarene testne metrike nalaze se u `outputs/local_model/model/model_metadata.json`, a usporedba modela u `outputs/local_model/metrics/model_comparison_results.csv`. Glavne slike za dokumentaciju nalaze se u `outputs/eda` i `outputs/local_model/figures`.

## API format

Primjer ulaznog zahtjeva nalazi se u `sample_request.json`. Endpoint podržava format:

```json
{
  "data": [
    {
      "Gender": "Male",
      "Age": 22,
      "Academic Pressure": 3,
      "Work Pressure": 0,
      "CGPA": 8.0,
      "Study Satisfaction": 3,
      "Job Satisfaction": 0,
      "Sleep Duration": "'5-6 hours'",
      "Dietary Habits": "Moderate",
      "Degree": "B.Tech",
      "Have you ever had suicidal thoughts ?": "No",
      "Work/Study Hours": 6,
      "Financial Stress": 2,
      "Family History of Mental Illness": "No"
    }
  ]
}
```

Primjer odgovora:

```json
[
  {
    "prediction": 0,
    "label": "No depression risk",
    "probability": 0.2345
  }
]
```
