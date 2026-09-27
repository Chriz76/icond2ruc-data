import os
import sys
import requests
import shutil
import json
import time
import glob
from datetime import datetime, timezone, timedelta

# Erlaubt den Import von process.py im selben Ordner
if "__file__" in globals():
    sys.path.append(os.path.dirname(__file__))
else:
    sys.path.append(os.path.abspath("."))

try:
    from process import AromeWindProcessor
except ImportError:
    pass  # Die Klasse 'AromeWindProcessor' ist in Colab bereits im RAM!

# --- KONFIGURATION ---
OUTPUT_DIR = "./output"
TEMP_GRIB_DIR = "./grib_temp"
FORECASTLENGTH = 24
API_VERSION = "1.1.0"
MAX_WEBP_COUNT = 50  # Maximal zu behaltende WebP-Dateien

# ROOT_FOLDER zeigt dorthin, wo clat.grib2 und clon.grib2 liegen
if "__file__" in globals():
    ROOT_FOLDER = os.path.dirname(__file__)
else:
    ROOT_FOLDER = os.path.abspath(".")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_GRIB_DIR, exist_ok=True)


def download_file(url, local_path):
    """Hilfsfunktion für echte Downloads mit Stream-Pufferung und 3 Retries"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=15, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"   ⚠️ Download-Versuch {attempt} fehlgeschlagen (Status: {response.status_code})")
        except Exception as e:
            print(f"   ⚠️ Download-Fehler bei Versuch {attempt}: {e}")

        if attempt < max_retries:
            wait_time = attempt * 2
            print(f"   ⏳ Warte {wait_time} Sekunden vor nächstem Versuch...")
            time.sleep(wait_time)

    return False


def process_single_hour(processor, target_time, f_hour, temp_dir=TEMP_GRIB_DIR, prefer_local=False):
    """
    Verarbeitet genau einen Schritt (Download optional).
    - Wenn `prefer_local` True ist, werden vorhandene Dateien in `temp_dir` verwendet und nicht gelöscht.
    - Wenn `prefer_local` False ist, werden fehlende Dateien heruntergeladen und anschließend (wie zuvor) gelöscht.
    Rückgabe: (success: bool, time_key: str)
    """
    valid_time = target_time + timedelta(hours=f_hour)
    time_key = valid_time.strftime('%Y%m%d_%H')
    png_filename = f"{time_key}Z.png"

    # URLs für U, V und VMAX (wie vorher)
    url_u = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/U_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT{f_hour:03d}H00M.grib2"
    url_v = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/V_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT{f_hour:03d}H00M.grib2"
    url_vmax = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/VMAX_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT{f_hour+1:03d}H00M.grib2"

    os.makedirs(temp_dir, exist_ok=True)
    u_path = os.path.join(temp_dir, f"u_{time_key}.grib2")
    v_path = os.path.join(temp_dir, f"v_{time_key}.grib2")
    vmax_path = os.path.join(temp_dir, f"vmax_{time_key}.grib2")

    have_local = os.path.exists(u_path) and os.path.exists(v_path) and os.path.exists(vmax_path)

    if prefer_local and have_local:
        print(f"   -> Verwende lokale GRIBs für {time_key} (prefer_local=True).")
    else:
        print(f"   -> Downloade Schritt +{f_hour}h...")
        ok_u = os.path.exists(u_path) or download_file(url_u, u_path)
        ok_v = os.path.exists(v_path) or download_file(url_v, v_path)
        ok_vmax = os.path.exists(vmax_path) or download_file(url_vmax, vmax_path)
        if not (ok_u and ok_v and ok_vmax):
            print(f"   ❌ Fehler beim Download von Schritt +{f_hour}h.")
            return False, time_key

    success = processor.process_step(u_path, v_path, vmax_path, time_key, png_filename)

    # Lösche nur, wenn prefer_local False (verhält sich ansonsten wie vorher: Dateien werden entfernt)
    if not prefer_local:
        for p in (u_path, v_path, vmax_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f"   ⚠️ Fehler beim Löschen von {p}: {e}")
    else:
        print(f"   → GRIBs für {time_key} bleiben erhalten (prefer_local=True).")

    if success:
        print(f"      ✅ Schritt +{f_hour}h erfolgreich prozessiert.")

    return success, time_key


def cleanup_old_webps(output_dir, max_keep=50):
    """Löscht ältere WebP-Dateien im Ausgabeordner basierend auf dem Dateinamen."""
    webp_files = glob.glob(os.path.join(output_dir, "*.webp"))
    
    if len(webp_files) > max_keep:
        print(f"\n🧹 Bereinige alte WebPs ({len(webp_files)} vorhanden, maximal {max_keep} erlaubt)...")
        # Alphabethische Sortierung nach Dateinamen (älteste Timestamps stehen vorne)
        webp_files.sort()
        
        # Alle Dateien bis auf die letzten max_keep (die neuesten) löschen
        files_to_delete = webp_files[:-max_keep]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"   🗑️ Gelöscht: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   ⚠️ Fehler beim Löschen von {file_path}: {e}")
        print(f"✅ Bereinigung abgeschlossen. Es verbleiben {max_keep} WebP-Dateien.")


def run_ruc_pipeline():
    # --- Ziel-Lauf direkt aus der Cloudflare-Übergabe auslesen ---
    dwd_run_env = os.environ.get("DWD_TARGET_RUN")
    
    if not dwd_run_env:
        print("❌ FEHLER: Umgebungsvariable 'DWD_TARGET_RUN' fehlt! Wurde die Action via Cloudflare gestartet?")
        sys.exit(1)

    # dwd_run_env sieht so aus: "2026-06-29T18:00"
    target_time = datetime.strptime(dwd_run_env, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    dwd_run_folder = target_time.strftime("%Y-%m-%dT%H:00")

    print(f"\n========================================================")
    print(f"START PIPELINE - Verarbeite DWD-Run: {dwd_run_folder}Z")
    print(f"========================================================")

    # Initialisiere den WindProcessor anhand deines korrekten Setups
    processor = WindProcessor(root_folder=ROOT_FOLDER, output_folder=OUTPUT_DIR, timeLineLength=FORECASTLENGTH + 5)
    processor.cluster_output_folder = os.path.join(OUTPUT_DIR, "grid_cluster")
    os.makedirs(processor.cluster_output_folder, exist_ok=True)

    detected_current_hour = target_time.strftime('%Y%m%d_%H')

    # Da Cloudflare steuert, laden wir stur alle Schritte (0 bis 24) für diesen Lauf
    missing_hours = list(range(FORECASTLENGTH + 1))

    print(f"👑 Server bereit! Starte Verarbeitung von {len(missing_hours)} Schritten...")

    for f_hour in missing_hours:
        # Verwende die neue helper-Funktion, Verhalten unverändert (default prefer_local=False)
        success, _ = process_single_hour(processor, target_time, f_hour, temp_dir=TEMP_GRIB_DIR, prefer_local=False)

    # =========================================================================
    # FINALES SPEICHERN & DYNAMISCHE ERSTELLUNG DER INDEX.JSON
    # =========================================================================
    processor.flush_json_to_disk()

    all_timestamps = []
    if processor.cluster_memory:
        first_cluster = next(iter(processor.cluster_memory.values()))
        if "timeline" in first_cluster:
            all_timestamps = list(first_cluster["timeline"].keys())

    if all_timestamps:
        index_path = os.path.join(OUTPUT_DIR, "index.json")
        sorted_timestamps = sorted(all_timestamps)

        if detected_current_hour and detected_current_hour in sorted_timestamps:
            current_hour = detected_current_hour
        else:
            if len(sorted_timestamps) >= FORECASTLENGTH + 5:
                current_hour = sorted_timestamps[len(sorted_timestamps) // 2]
            else:
                current_hour = sorted_timestamps[-1]

        index_data = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "available_timestamps": sorted_timestamps,
            "current_hour": current_hour,
            "api_version": API_VERSION
        }

        print(f"\n📝 [Main Program] Generiere {index_path}...")
        print(f"   -> Verfügbare Schritte im Frontend: {len(sorted_timestamps)}")
        print(f"   -> Exakt detektierter Standard-Fokus (current_hour): {current_hour}")

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print("✅ index.json erfolgreich aktualisiert!")
    else:
        print("⚠️ Warnung: Keine Daten-Timestamps für index.json gefunden.")

    # Alte WebP-Dateien nach Dateinamen-Sortierung bereinigen
    cleanup_old_webps(OUTPUT_DIR, max_keep=MAX_WEBP_COUNT)

    if os.path.exists(TEMP_GRIB_DIR):
        shutil.rmtree(TEMP_GRIB_DIR)
        print("🧹 Temporärer Download-Ordner wurde vollständig bereinigt!")

    print("\n🎉 PIPELINE ERFOLGREICH BEENDET!")


# --- STEUERUNG BEIM LADEN/AUSFÜHREN ---
if __name__ == "__main__":
    # Prüft, ob das Skript über ein Terminal/GitHub Actions gestartet wurde (sys.ps1 existiert nur in Shells nicht)
    is_interactive = hasattr(sys, "ps1") or "ipykernel" in sys.modules

    if not is_interactive:
        # Auf GitHub Actions: Automatisch ausführen (stürzt ab, falls DWD_TARGET_RUN fehlt)
        run_ruc_pipeline()
    else:
        # In Colab/Jupyter: Nur in den Speicher laden, Nichts tun
        print("📦 Modul erfolgreich in den Colab-Speicher geladen (Pipeline nicht gestartet).")