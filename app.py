import customtkinter as ctk
import subprocess
import threading
import sys
import json

ctk.set_appearance_mode("dark")  

class NBAPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NBA Predictor IA - Panel de Control")
        self.geometry("750x700")
        self.resizable(False, False)

        self.label_title = ctk.CTkLabel(self, text="NBA Predictor IA", font=("Roboto", 28, "bold"))
        self.label_title.pack(pady=(20, 5))
        
        self.label_subtitle = ctk.CTkLabel(self, text="Analisis de Machine Learning en tiempo real", font=("Roboto", 14), text_color="gray")
        self.label_subtitle.pack(pady=(0, 15))

        self.btn_run = ctk.CTkButton(
            self, text="Generar Pronosticos de Hoy", font=("Roboto", 16, "bold"),
            height=40, fg_color="#C9082A", hover_color="#9B0620",
            command=self.start_prediction_thread
        )
        self.btn_run.pack(pady=10)

        # Contenedor con Scroll para las tarjetas
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=650, height=450, fg_color="transparent")
        self.scrollable_frame.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(self.scrollable_frame, text="Sistema listo. Presiona el boton para iniciar.", font=("Roboto", 14))
        self.status_label.pack(pady=50)

    def start_prediction_thread(self):
        self.btn_run.configure(state="disabled", text="Calculando...")
        
        # Limpiar resultados anteriores
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.status_label = ctk.CTkLabel(self.scrollable_frame, text="Consultando API de la NBA e inferiendo redes neuronales...\nEsto puede tardar unos segundos.", font=("Roboto", 14))
        self.status_label.pack(pady=50)
        
        thread = threading.Thread(target=self.run_predictions)
        thread.start()

    def run_predictions(self):
        try:
            process = subprocess.run([sys.executable, "src/predict.py"], capture_output=True, text=True)
            output = process.stdout
            
            # Extraer solo el JSON limpio ignorando advertencias de TensorFlow
            if "---JSON_START---" in output and "---JSON_END---" in output:
                json_str = output.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
                games_data = json.loads(json_str)
                
                # Actualizar la interfaz (debe hacerse en el hilo principal de manera indirecta)
                self.after(0, self.render_cards, games_data)
            else:
                self.after(0, self.show_error, "No hay partidos hoy o error en la API.")
                
        except Exception as e:
            self.after(0, self.show_error, f"Fallo al ejecutar: {e}")
            
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="Generar Pronosticos de Hoy"))

    def show_error(self, message):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        error_lbl = ctk.CTkLabel(self.scrollable_frame, text=message, text_color="red", font=("Roboto", 14))
        error_lbl.pack(pady=20)

    def render_cards(self, games_data):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        if not games_data:
            lbl = ctk.CTkLabel(self.scrollable_frame, text="No hay partidos programados para hoy.", font=("Roboto", 16))
            lbl.pack(pady=50)
            return

        for game in games_data:
            # Tarjeta principal del partido
            card = ctk.CTkFrame(self.scrollable_frame, fg_color="#1E1E1E", corner_radius=10, border_color="#333333", border_width=1)
            card.pack(pady=10, padx=10, fill="x")

            # Titulo del Encuentro (Ej. ATL @ CLE)
            match_title = f"{game['away_team']} @ {game['home_team']}"
            title_lbl = ctk.CTkLabel(card, text=match_title, font=("Roboto", 24, "bold"))
            title_lbl.pack(pady=(10, 0))

            # Probabilidades
            prob_text = f"{game['away_name']} ({game['away_prob']}%)  vs  {game['home_name']} ({game['home_prob']}%)"
            prob_lbl = ctk.CTkLabel(card, text=prob_text, font=("Roboto", 14), text_color="#A0A0A0")
            prob_lbl.pack(pady=2)

            # Pronostico Ganador
            winner_lbl = ctk.CTkLabel(card, text=f"GANA {game['predicted_winner'].upper()}", font=("Roboto", 18, "bold"), text_color="#C9082A")
            winner_lbl.pack(pady=(5, 10))

            # Seccion de Lesiones
            home_inj_str = ", ".join(game['home_injuries']) if game['home_injuries'] else "Plantel Completo"
            away_inj_str = ", ".join(game['away_injuries']) if game['away_injuries'] else "Plantel Completo"
            
            inj_frame = ctk.CTkFrame(card, fg_color="transparent")
            inj_frame.pack(fill="x", padx=15, pady=(0, 10))
            
            away_inj_lbl = ctk.CTkLabel(inj_frame, text=f"Bajas Visita ({game['away_team']}): {away_inj_str}", font=("Roboto", 11), text_color="gray", justify="left", wraplength=550)
            away_inj_lbl.pack(anchor="w")
            
            home_inj_lbl = ctk.CTkLabel(inj_frame, text=f"Bajas Local ({game['home_team']}): {home_inj_str}", font=("Roboto", 11), text_color="gray", justify="left", wraplength=550)
            home_inj_lbl.pack(anchor="w")

if __name__ == "__main__":
    app = NBAPredictorApp()
    app.mainloop()