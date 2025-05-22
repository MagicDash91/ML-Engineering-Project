
# Crop Analysis with CSV Input, Bar Chart, and Word Cloud Visualization

This project provides a simple data analysis tool for crop-related symptoms using a CSV input file. The goal is to analyze and visualize crop health issues by country and plant type, and generate a word cloud of symptom descriptions filtered by plant.

## Features

- Upload a CSV file containing crop data.
- Select columns: `Country`, `Plant`, and `Symptomp` (textual symptom descriptions).
- Generate bar charts showing the top 10 countries and top 10 plants by record count.
- Generate a word cloud based on the `Symptomp` text filtered by a selected plant.
- Visualization helps in quick insight into crop health issues geographically and by plant type.

## Input CSV Format

Your CSV file should contain at least the following columns:

- `Country` (string) — Country names.
- `Plant` (string) — Plant or crop names.
- `Symptomp` (string) — Text descriptions of plant symptoms/issues.

Example row:

| Country  | Plant  | Symptomp                       |
|----------|--------|-------------------------------|
| USA      | Tomato | My plant is died yesterday     |
| Brazil   | Corn   | Why my tomato isn't growing    |

## How to Run

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/crop-analysis.git
    cd crop-analysis
    ```

2. Create and activate a virtual environment (recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Run the main script or FastAPI server (depending on your implementation):

    ```
    python main.py
    # or
    uvicorn app:app --reload
    ```

5. Upload your CSV file, select the relevant columns (`Country`, `Plant`, `Symptomp`), and visualize:

- Bar charts for top 10 countries and plants by record count.
- Word cloud filtered by selected plant's symptom texts.

## Dependencies

- pandas
- polars
- fastapi
- matplotlib
- seaborn
- wordcloud
- pillow
- google-generativeai (optional, if you integrate with Google AI)
- python-dotenv

Make sure to have these installed via the requirements file or manually.

## Notes

- The `Symptomp` column contains textual descriptions. The word cloud visualizes the most frequent words appearing in symptoms of the selected plant.
- Bar charts only show the top 10 entries for clarity.
- You can customize the word cloud and bar chart styles within the script as needed.

---

Feel free to extend or improve this project with additional visualization or predictive modeling.

---

### Contact

Created by Michael.
