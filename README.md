<h1 align="center">
  GLA Pay-Gap Insights 📊
</h1>

<p align="center">
  <em>Streamlit dashboard that turns three GLA workbooks into share-worthy insight &nbsp;―&nbsp; disability, gender and ethnicity pay gaps in one click.</em>
</p>

<div align="center">
  <a href="https://streamlit.io"><img alt="Built with Streamlit"
    src="https://img.shields.io/badge/built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white"></a>
  <a href="https://github.com/YOUR_HANDLE/gla-pay-gap-insights/blob/main/LICENSE">
    <img alt="license: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="Python 3.9 +" src="https://img.shields.io/badge/python-3.9%2B-blue">
</div>

---

### ✨ Why this exists
London’s GLA publishes pay-gap spreadsheets every year—but they’re locked away in multi-sheet, human-oriented tables.  
This repo **parses those files on the fly** and serves an interactive web app with:

* one-click switching between **Disability, Gender, Ethnicity**
* **Hourly-Pay** **&** **Pay-Gap** views
* auto-computed insights (start/end, Δ, %Δ, σ-volatility, trend arrow)
* highlight call-outs for the biggest ↑ / ↓ shifts
* a quick-and-dirty **next-year forecast** per category

> Want a polished slide in < 60 s?  
> Drop the XLSX files next to the script, run `streamlit run app.py`, grab the screenshot.

---

### 🏁 Quick-start

```bash
# 1. clone
git clone https://github.com/YOUR_HANDLE/gla-pay-gap-insights.git
cd gla-pay-gap-insights

# 2. drop the raw workbooks here (exact filenames):
#    ├── Disability pay gap data tables 2021-2024.xlsx
#    ├── Gender pay gap data tables 2017-2024.xlsx
#    └── Ethnicity pay gap data tables 2017-2024.xlsx

# 3. install deps
pip install -r requirements.txt   # python 3.9+

# 4. 🚀
streamlit run app.py
