import customtkinter as ctk
import subprocess
import threading
import sys
import json
import os

ctk.set_appearance_mode("dark")  

class NBAPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NBA Predictor IA - Panel de Control")
        self.geometry("750x700")
        self.resizable(False, False)

        self.label_title = ctk.CTkLabel(self, text="NBA Predictor IA", font=("Roboto", 28, "bold"))
        self.label_title.pack(pady=(20, 0))
        
        # Insignia de precision (Badge)
        self.badge_acc = ctk.CTkLabel(self, text="Precision actual: --%", font=("Roboto", 13, "bold"), text_color="gray")
        self.badge_acc.pack(pady=(2, 5))
        self.update_accuracy_badge() # Llamada inicial para leer el archivo si existe

        self.label_subtitle = ctk.CTkLabel(self, text="Analisis de Machine Learning en tiempo real", font=("Roboto", 14), text_color="gray")
        self.label_subtitle.pack(pady=(0, 15))

        # --- NUEVO: Contenedor para multiples botones ---
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=10)

        # Boton 1: Prediccion Rapida (Rojo)
        self.btn_run = ctk.CTkButton(
            self.button_frame, text="Ver Pronosticos", font=("Roboto", 15, "bold"),
            height=40, fg_color="#C9082A", hover_color="#9B0620",
            command=self.start_prediction_thread
        )
        self.btn_run.grid(row=0, column=0, padx=10)

        # Boton 2: Ejecutar Pipeline Completo (Gris)
        self.btn_pipeline = ctk.CTkButton(
            self.button_frame, text="Ejecutar Pipeline Completo", font=("Roboto", 15),
            height=40, fg_color="#404040", hover_color="#2D2D2D",
            command=self.start_pipeline_thread
        )
        self.btn_pipeline.grid(row=0, column=1, padx=10)
        # ------------------------------------------------

        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=650, height=450, fg_color="transparent")
        self.scrollable_frame.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(self.scrollable_frame, text="Sistema listo.\nUsa el Pipeline una vez al dia para actualizar bases de datos.", font=("Roboto", 14))
        self.status_label.pack(pady=50)

    def update_accuracy_badge(self):
        """Lee el archivo accuracy.txt y actualiza la etiqueta en la interfaz."""
        try:
            acc_path = os.path.join("models", "accuracy.txt")
            if os.path.exists(acc_path):
                with open(acc_path, "r") as f:
                    acc_value = f.read().strip()
                    self.badge_acc.configure(text=f"Precision de la Red Neuronal: {acc_value}%", text_color="#2FA572")
        except Exception:
            pass

    # --- LOGICA DEL BOTON 1 (Solo Predicciones) ---
    def start_prediction_thread(self):
        self.btn_run.configure(state="disabled")
        self.btn_pipeline.configure(state="disabled")
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.status_label = ctk.CTkLabel(self.scrollable_frame, text="Consultando probabilidades...\nEsto tomara unos segundos.", font=("Roboto", 14))
        self.status_label.pack(pady=50)
        
        thread = threading.Thread(target=self.run_predictions)
        thread.start()

    def run_predictions(self):
        try:
            python_bin = sys.executable.replace("pythonw.exe", "python.exe")
            process = subprocess.run([python_bin, "src/predict.py"], capture_output=True, text=True, encoding='utf-8')
            output = process.stdout
            
            if "---JSON_START---" in output and "---JSON_END---" in output:
                json_str = output.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
                games_data = json.loads(json_str)
                self.after(0, self.render_cards, games_data)
            else:
                error_msg = process.stderr if process.stderr else "Sin datos. Asegurate de correr el Pipeline primero."
                self.after(0, self.show_error, f"Error:\n{error_msg}")
                
        except Exception as e:
            self.after(0, self.show_error, f"Fallo al predecir: {e}")
            
        finally:
            self.after(0, self.enable_buttons)

    # --- NUEVA LOGICA DEL BOTON 2 (Pipeline Completo) ---
    def start_pipeline_thread(self):
        self.btn_run.configure(state="disabled")
        self.btn_pipeline.configure(state="disabled", text="Procesando datos...")
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.status_label = ctk.CTkLabel(self.scrollable_frame, text="Ejecutando orquestador maestro (run_pipeline.py)...\nDescargando resultados, lesiones y evaluando red neuronal.\nPor favor, espera...", font=("Roboto", 14))
        self.status_label.pack(pady=50)
        
        thread = threading.Thread(target=self.run_pipeline_process)
        thread.start()

    def run_pipeline_process(self):
        try:
            python_bin = sys.executable.replace("pythonw.exe", "python.exe")
            process = subprocess.run([python_bin, "run_pipeline.py"], capture_output=True, text=True, encoding='utf-8')
            
            if process.returncode == 0:
                # Actualiza la precision si el pipeline incluyo el entrenamiento
                self.after(0, self.update_accuracy_badge)
                # Inmediatamente despues, encadena la busqueda de pronosticos
                self.after(0, self.run_predictions)
            else:
                self.after(0, self.show_error, f"Fallo el Pipeline:\n{process.stderr}")
                
        except Exception as e:
            self.after(0, self.show_error, f"Fallo de ejecucion: {e}")
        finally:
            self.after(0, self.enable_buttons)

    def enable_buttons(self):
        self.btn_run.configure(state="normal", text="Ver Pronosticos")
        self.btn_pipeline.configure(state="normal", text="Ejecutar Pipeline Completo")

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
            card = ctk.CTkFrame(self.scrollable_frame, fg_color="#1E1E1E", corner_radius=10, border_color="#333333", border_width=1)
            card.pack(pady=10, padx=10, fill="x")

            match_title = f"{game['away_team']} @ {game['home_team']}"
            title_lbl = ctk.CTkLabel(card, text=match_title, font=("Roboto", 24, "bold"))
            title_lbl.pack(pady=(10, 0))

            prob_text = f"{game['away_name']} ({game['away_prob']}%)  vs  {game['home_name']} ({game['home_prob']}%)"
            prob_lbl = ctk.CTkLabel(card, text=prob_text, font=("Roboto", 14), text_color="#A0A0A0")
            prob_lbl.pack(pady=2)

            winner_lbl = ctk.CTkLabel(card, text=f"GANA {game['predicted_winner'].upper()}", font=("Roboto", 18, "bold"), text_color="#C9082A")
            winner_lbl.pack(pady=(5, 10))

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