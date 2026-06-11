import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_player_dataset(output_path, num_players=600):
    # Lists of realistic values
    first_names = [
        "Erling", "Kylian", "Mohamed", "Kevin", "Jude", "Bukayo", "Martin", "Harry", "Bruno", "Marcus",
        "Declan", "Rodri", "Virgil", "William", "Ruben", "Trent", "Reece", "Luka", "Antoine", "Robert",
        "Vinicius", "Pedri", "Gavi", "Federico", "Aurélien", "Eduardo", "Lautaro", "Victor", "Rafael", "Khvicha",
        "Paulo", "Romelu", "Dušan", "Teun", "Hakan", "Nicolò", "Jamal", "Leroy", "Florian", "Harry",
        "Thomas", "Joshua", "Granit", "Jeremie", "Alex", "Son", "Dejan", "James", "Alexander", "Mohamed",
        "Bernardo", "Phil", "Jack", "Alexis", "Dominik", "Luis", "Darwin", "Alisson", "Ederson", "Mike",
        "Marc-André", "Jan", "Thibaut", "Gianluigi", "Diogo", "Bruno", "Casemiro", "Christian", "Rasmus", "Alejandro"
    ]
    
    last_names = [
        "Haaland", "Mbappé", "Salah", "De Bruyne", "Bellingham", "Saka", "Ødegaard", "Kane", "Fernandes", "Rashford",
        "Rice", "Hernández", "van Dijk", "Saliba", "Dias", "Alexander-Arnold", "James", "Modrić", "Griezmann", "Lewandowski",
        "Júnior", "González", "Páez", "Valverde", "Tchouaméni", "Camavinga", "Martínez", "Osimhen", "Leão", "Kvaratskhelia",
        "Dybala", "Lukaku", "Vlahović", "Koopmeiners", "Çalhanoğlu", "Barella", "Musiala", "Sané", "Wirtz", "Kane",
        "Müller", "Kimmich", "Xhaka", "Frimpong", "Grimaldo", "Min-son", "Kulusevski", "Maddison", "Isak", "Kudus",
        "Silva", "Foden", "Grealish", "Mac Allister", "Szoboszlai", "Díaz", "Núñez", "Becker", "Moraes", "Maignan",
        "ter Stegen", "Oblak", "Courtois", "Donnarumma", "Jota", "Guimarães", "Casemiro", "Eriksen", "Højlund", "Garnacho"
    ]
    
    nationalities = [
        "Norway", "France", "Egypt", "Belgium", "England", "Portugal", "Netherlands", "Brazil", "Spain", "Germany",
        "Argentina", "Italy", "Uruguay", "Nigeria", "Georgia", "Poland", "Croatia", "Switzerland", "Denmark", "Sweden",
        "Scotland", "Wales", "Ireland", "Senegal", "Morocco", "Algeria", "Ivory Coast", "Cameroon", "Japan", "South Korea"
    ]
    
    leagues_clubs = {
        "English Premier League": ["Man City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham", "Chelsea", "Man United", "Newcastle", "West Ham", "Brighton"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Athletic Club", "Girona", "Real Betis", "Villarreal", "Sevilla", "Valencia"],
        "Serie A": ["Inter", "Milan", "Juventus", "Bologna", "Roma", "Atalanta", "Lazio", "Napoli", "Fiorentina", "Torino"],
        "Bundesliga": ["Leverkusen", "Bayern Munich", "Stuttgart", "Dortmund", "Leipzig", "Frankfurt", "Freiburg", "Hoffenheim", "Werder Bremen", "Wolfsburg"],
        "Ligue 1": ["PSG", "Monaco", "Brest", "Lille", "Nice", "Lens", "Marseille", "Lyon", "Rennes", "Reims"]
    }
    
    positions = ["Forward", "Winger", "Midfielder", "Defender"]
    
    players = []
    
    # Track unique player names to avoid duplicates
    generated_names = set()
    
    for i in range(num_players):
        # Generate unique name
        while True:
            first = np.random.choice(first_names)
            last = np.random.choice(last_names)
            name = f"{first} {last}"
            if name not in generated_names:
                generated_names.add(name)
                break
        
        # Demographic details
        age = int(np.random.randint(18, 37))
        nationality = np.random.choice(nationalities)
        league = np.random.choice(list(leagues_clubs.keys()))
        club = np.random.choice(leagues_clubs[league])
        position = np.random.choice(positions)
        
        # Minutes played
        minutes_played = float(np.random.randint(400, 3400))
        games_played_approx = minutes_played / 90.0
        
        # Stat generation based on position
        if position == "Forward":
            # Focus on goals, low defensive stats, moderate passing
            goals_per_90 = np.random.uniform(0.35, 0.85)
            assists_per_90 = np.random.uniform(0.05, 0.25)
            passing_accuracy = np.random.uniform(70.0, 83.0)
            tackles_per_90 = np.random.uniform(0.1, 0.8)
            interceptions_per_90 = np.random.uniform(0.05, 0.5)
            dribbles_per_90 = np.random.uniform(0.5, 2.2)
            
        elif position == "Winger":
            # High assists, high dribbles, moderate goals
            goals_per_90 = np.random.uniform(0.2, 0.55)
            assists_per_90 = np.random.uniform(0.18, 0.5)
            passing_accuracy = np.random.uniform(74.0, 85.0)
            tackles_per_90 = np.random.uniform(0.3, 1.4)
            interceptions_per_90 = np.random.uniform(0.15, 0.8)
            dribbles_per_90 = np.random.uniform(1.8, 4.2)
            
        elif position == "Midfielder":
            # We sub-divide into "Playmaker" style (high passing/assists) and "Defensive" style (high tackles/interceptions)
            mid_type = np.random.choice(["playmaker", "defensive", "box-to-box"])
            if mid_type == "playmaker":
                goals_per_90 = np.random.uniform(0.05, 0.25)
                assists_per_90 = np.random.uniform(0.22, 0.55)
                passing_accuracy = np.random.uniform(84.0, 93.0)
                tackles_per_90 = np.random.uniform(0.4, 1.6)
                interceptions_per_90 = np.random.uniform(0.4, 1.5)
                dribbles_per_90 = np.random.uniform(1.0, 2.8)
            elif mid_type == "defensive":
                goals_per_90 = np.random.uniform(0.01, 0.1)
                assists_per_90 = np.random.uniform(0.03, 0.15)
                passing_accuracy = np.random.uniform(82.0, 91.0)
                tackles_per_90 = np.random.uniform(2.0, 4.4)
                interceptions_per_90 = np.random.uniform(1.5, 3.4)
                dribbles_per_90 = np.random.uniform(0.3, 1.5)
            else: # Box-to-box
                goals_per_90 = np.random.uniform(0.08, 0.3)
                assists_per_90 = np.random.uniform(0.08, 0.3)
                passing_accuracy = np.random.uniform(80.0, 88.0)
                tackles_per_90 = np.random.uniform(1.4, 2.8)
                interceptions_per_90 = np.random.uniform(1.0, 2.5)
                dribbles_per_90 = np.random.uniform(0.8, 2.2)
                
        else: # Defender
            # We sub-divide into "Center Back" (high defensive, low dribbles, low assists) and "Fullback" (moderate assists, moderate dribbles, high tackles)
            def_type = np.random.choice(["cb", "fb"])
            if def_type == "cb":
                goals_per_90 = np.random.uniform(0.0, 0.08)
                assists_per_90 = np.random.uniform(0.0, 0.05)
                passing_accuracy = np.random.uniform(83.0, 92.0)
                tackles_per_90 = np.random.uniform(1.8, 3.6)
                interceptions_per_90 = np.random.uniform(1.6, 3.8)
                dribbles_per_90 = np.random.uniform(0.05, 0.5)
            else: # Fullback / Wingback
                goals_per_90 = np.random.uniform(0.01, 0.12)
                assists_per_90 = np.random.uniform(0.1, 0.32)
                passing_accuracy = np.random.uniform(77.0, 86.0)
                tackles_per_90 = np.random.uniform(1.6, 3.2)
                interceptions_per_90 = np.random.uniform(1.1, 2.6)
                dribbles_per_90 = np.random.uniform(0.8, 2.2)

        # Scale stats to season totals
        goals = max(0, int(round(goals_per_90 * games_played_approx)))
        assists = max(0, int(round(assists_per_90 * games_played_approx)))
        tackles = max(0.0, round(tackles_per_90 * games_played_approx, 1))
        interceptions = max(0.0, round(interceptions_per_90 * games_played_approx, 1))
        dribbles = max(0.0, round(dribbles_per_90 * games_played_approx, 1))
        passing_accuracy = round(passing_accuracy, 1)

        # Calculate a realistic market value (in Millions of Euros)
        # Value base based on performance metrics scaled by minutes
        val_base = (
            goals * 3.5 + 
            assists * 2.5 + 
            (passing_accuracy - 70.0) * 0.4 + 
            (tackles + interceptions) * 0.8 + 
            dribbles * 1.0
        )
        
        # Age multiplier (prime is 23-28)
        if 23 <= age <= 28:
            age_mult = np.random.uniform(1.5, 2.8)
        elif age < 23:
            # Young prospects with high potential multiplier
            age_mult = np.random.uniform(1.2, 2.5)
        else:
            # Decline for older players
            age_mult = max(0.1, np.random.uniform(0.2, 0.8) - 0.05 * (age - 29))
            
        # League multiplier
        league_mults = {
            "English Premier League": 1.6,
            "La Liga": 1.25,
            "Bundesliga": 1.15,
            "Serie A": 1.1,
            "Ligue 1": 0.85
        }
        league_mult = league_mults[league]
        
        market_value = round(max(0.5, val_base * age_mult * league_mult * np.random.uniform(0.8, 1.2)), 1)
        
        players.append({
            "player_name": name,
            "age": age,
            "nationality": nationality,
            "club": club,
            "league": league,
            "position": position,
            "market_value": market_value,
            "goals": float(goals),
            "assists": float(assists),
            "minutes_played": minutes_played,
            "passing_accuracy": passing_accuracy,
            "tackles": tackles,
            "interceptions": interceptions,
            "dribbles": dribbles
        })
    
    df = pd.DataFrame(players)
    
    # Inject 2-3% missing values for data cleaning demonstration
    # We will inject some NaNs in passing_accuracy, tackles, and interceptions
    num_nan_pass = int(num_players * 0.02)
    num_nan_tackles = int(num_players * 0.02)
    num_nan_interceptions = int(num_players * 0.02)
    
    nan_pass_indices = np.random.choice(df.index, num_nan_pass, replace=False)
    nan_tackles_indices = np.random.choice(df.index, num_nan_tackles, replace=False)
    nan_interceptions_indices = np.random.choice(df.index, num_nan_interceptions, replace=False)
    
    df.loc[nan_pass_indices, "passing_accuracy"] = np.nan
    df.loc[nan_tackles_indices, "tackles"] = np.nan
    df.loc[nan_interceptions_indices, "interceptions"] = np.nan
    
    # Ensure directory exists and write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_players} players and saved to {output_path}")

if __name__ == "__main__":
    generate_player_dataset("data/players.csv")
