import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_injuries():
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print("Obteniendo datos de lesiones desde CBS Sports...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error de red al intentar acceder a la pagina: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontramos los contenedores principales que agrupan a cada equipo de forma individual
    team_containers = soup.find_all('div', class_='TableBase')
    
    injury_data = []
    
    for container in team_containers:
        # 1. Extraer el nombre del equipo confinado solo a este contenedor
        team_tag = container.find(class_='TeamName')
        if not team_tag:
            continue
            
        team_name = team_tag.get_text(strip=True)
        
        # 2. Extraer los jugadores solo de la tabla dentro de este mismo contenedor
        rows = container.select('tbody tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                # Limpieza del nombre para evitar textos empalmados como "J. TatumJayson Tatum"
                player_cell = cols[0]
                long_name_tag = player_cell.find(class_='CellPlayerName--long')
                
                if long_name_tag and long_name_tag.find('a'):
                    player_name = long_name_tag.find('a').get_text(strip=True)
                else:
                    player_name = player_cell.get_text(strip=True)
                    
                position = cols[1].get_text(strip=True)
                
                # Tomamos la ultima columna disponible en la fila para el detalle de la lesion
                status = cols[-1].get_text(strip=True)
                
                injury_data.append({
                    'TEAM': team_name,
                    'PLAYER': player_name,
                    'POSITION': position,
                    'STATUS': status
                })
                
    return pd.DataFrame(injury_data)

def main():
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "current_injuries.csv")
    
    df = scrape_injuries()
    
    if df is not None and not df.empty:
        df.to_csv(output_file, index=False)
        print(f"Reporte de lesiones guardado en: {output_file}")
        print(f"Total de jugadores lesionados registrados: {len(df)}")
    else:
        print("No se encontraron datos de lesiones o la estructura de la pagina cambio.")

if __name__ == "__main__":
    main()