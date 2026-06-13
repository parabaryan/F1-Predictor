import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.jolpi.ca/ergast/f1"

def fetch_race_results(start_year=2000, end_year=2024):
    all_results = []
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        response = requests.get(f"{BASE_URL}/{year}/results.json?limit=500")
        
        if response.status_code != 200:
            print(f"Failed {year}")
            continue
            
        races = response.json()["MRData"]["RaceTable"]["Races"]
        
        for race in races:
            for result in race["Results"]:
                all_results.append({
                    "year": int(year),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit": race["Circuit"]["circuitId"],
                    "driver": result["Driver"]["driverId"],
                    "constructor": result["Constructor"]["constructorId"],
                    "grid": int(result["grid"]),
                    "position": int(result["position"]) if result["position"].isdigit() else 20,
                    "points": float(result["points"]),
                    "status": result["status"],
                    "laps": int(result["laps"]),
                })
        
        time.sleep(0.3)  # be nice to the API
    
    df = pd.DataFrame(all_results)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/race_results.csv", index=False)
    print(f"Saved {len(df)} rows to data/race_results.csv")
    return df

def fetch_qualifying(start_year=2000, end_year=2024):
    all_quali = []
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching qualifying {year}...")
        response = requests.get(f"{BASE_URL}/{year}/qualifying.json?limit=500")
        
        if response.status_code != 200:
            continue
            
        races = response.json()["MRData"]["RaceTable"]["Races"]
        
        for race in races:
            for result in race["QualifyingResults"]:
                all_quali.append({
                    "year": int(year),
                    "round": int(race["round"]),
                    "circuit": race["Circuit"]["circuitId"],
                    "driver": result["Driver"]["driverId"],
                    "constructor": result["Constructor"]["constructorId"],
                    "quali_position": int(result["position"]),
                })
        
        time.sleep(0.3)
    
    df = pd.DataFrame(all_quali)
    df.to_csv("data/qualifying.csv", index=False)
    print(f"Saved {len(df)} rows to data/qualifying.csv")
    return df

def fetch_driver_standings(start_year=2000, end_year=2024):
    all_standings = []
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching standings {year}...")
        response = requests.get(f"{BASE_URL}/{year}/driverStandings.json")
        
        if response.status_code != 200:
            continue
            
        standings = response.json()["MRData"]["StandingsTable"]["StandingsLists"]
        
        if not standings:
            continue
            
        for entry in standings[0]["DriverStandings"]:
            all_standings.append({
                "year": int(year),
                "driver": entry["Driver"]["driverId"],
                "championship_points": float(entry["points"]),
                "championship_wins": int(entry["wins"]),   
                "championship_position": int(pos) if str(pos := entry.get("positionText", entry.get("position", "0"))).isdigit() else 0,
                })
        
        time.sleep(0.3)
    
    df = pd.DataFrame(all_standings)
    df.to_csv("data/standings.csv", index=False)
    print(f"Saved {len(df)} rows to data/standings.csv")
    return df

if __name__ == "__main__":
    print("=== Fetching F1 Data (2000-2024) ===\n")
    fetch_race_results()
    fetch_qualifying()
    fetch_driver_standings()
    print("\nDone! All data saved to /data")