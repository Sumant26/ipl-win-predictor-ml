# 🏏 IPL Win Predictor

Ball-by-ball win probability for IPL matches, built from raw Cricsheet JSON — with win-probability "worm" charts and over-by-over commentary like *"CSK 68% to win"*.

**Skills:** JSON parsing · event-data feature engineering · data leakage (and how to catch it) · time-based splits · probability models · calibration · log loss / Brier score · monotonic constraints

**Time:** two to three weeks · **Level:** intermediate

## Run it
Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → `ipl_win_predictor.ipynb`, then run top to bottom. It downloads ~1,100 matches from [Cricsheet](https://cricsheet.org/downloads/) automatically.

## What's inside
1. Parse every match (innings → overs → deliveries) into a matches table and a balls table
2. Explore scoring trends by season and by over
3. Build match-state features after every ball using running totals
4. Train a chase model and a first-innings model on older seasons, test on the last 3
5. Show how **leakage** inflates scores (a future-information feature, and a random row split)
6. Check probabilities are honest with a calibration plot
7. Draw the win-probability worm for any match and print over-by-over commentary

## Ideas to extend
Venue averages, team form, recent momentum, a pre-match predictor, a Streamlit app.
# ipl-win-predictor-ml
