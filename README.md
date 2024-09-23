# 📊 Streamlit Habibie Tennis Project

Welcome to the **Streamlit Habibie Tennis Project**, a useful tool to understand ATP tennis players' performance using Elo ratings. This project leverages match data to dynamically track player performance across different surfaces. 

## 📖 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [App Dashboard](#app-dashboard)
- [Web Scraping](#web-scraping)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎾 Introduction

This project implements Elo ratings for ATP tennis players using historical match data. Elo ratings are calculated separately for each surface (grass, clay, and hard), with an overall rating (`Elo_ALL`). The analysis helps predict future match outcomes, track player progression, and visualize trends across surfaces and time.

## ✨ Features
- **Dynamic Elo Rating Calculations**: Separate Elo ratings for each surface and combined Elo ratings.
- **Expected Outcome Calculation**: Predict match outcomes based on Elo ratings, including head-to-head adjustments.
- **Visualization**: Elo rating trends displayed through line graphs for individual players and player comparisons.
- **Data Dashboard**: Interactive web-based dashboard using Dash for viewing player statistics and match history.
- **Web Scraping**: Automated script to scrape and collect ATP player information.

## ⚙️ Installation
To set up and run the project locally:
1. Clone this repository:
    ```bash
    git clone https://github.com/kaalvoetranger-88/st-habibies-bets.git
    ```
2. Navigate to the project directory:
    ```bash
    cd tennis-app
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Sources
This project uses ATP tennis match and player data from various sources:
- [Jeff Sackmann’s Tennis ATP Data](https://github.com/JeffSackmann/tennis_atp) for player dataframe
- [Tennis Results](http://tennis-data.co.uk/index.php) for matches dataframe
- Wikipedia for player images and biography.

## 🚀 Usage
1. **Data Preprocessing**: Load match datasets and append them into a unified DataFrame for analysis.
2. **Elo Rating Calculation**: Run scripts to calculate and update Elo ratings for each surface and overall.
3. **Prediction Testing**: Use the Elo system to test match predictions, calculate accuracy, and log loss.

To start the project, run the following:
```bash
python main.py
```

## 🖥️ App Dashboard
The project includes a web-based dashboard built using **Streamlit**. The dashboard allows users to:
- Select players and view their Elo history and match results.
- Compare Elo ratings of two players in interactive graphs.
- Display player statistics such as win percentages and Elo trends.
- See expected outcomes

### Running the Webapp
To launch the app with Streamlit, use:
```bash
streamlit run main.py
```

## 🤖 Web Scraping
An automated web scraping function collects player photo and 1st paragraph from their wikipedia page. The script uses requests and BeautifulSoup.

## 📈 Results and Visualizations
View graphs showing player ATP rankings and Elo ratings over time.
Prediction Accuracy: Detailed accuracy analysis of match outcome predictions based on Elo ratings.
Simulation tools allow you to also save the predictions to .csv

Sample visualizations: add by end Sept 2024!

## 🤝 Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork this repository.
Create a new branch for your feature:
```bash
git checkout -b feature-name
```
Commit your changes:
```bash
git commit -m "Add new feature"
```
Push the changes:
```bash
git push origin feature-name
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact
For any questions or suggestions, feel free to reach out:

GitHub: kaalvoetranger-88/
Email: kaalvoetranger@gmail.com
