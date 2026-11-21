export default {
  async scheduled(event, env, ctx) {
    const owner = "Chriz76";
    const repo = "icond2ruc-data";
    const pipelineBranch = "main"; // Der Branch, auf dem deine PNGs liegen (z.B. main oder gh-pages)
    const dataBranch = "gh-pages";
    const workflowFileName = "sync.yml";

    const githubHeaders = {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${env.GH_PAT}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "Cloudflare-Worker"
    };
    
    // ---------------------------------------------------------
    // SCHRITT 1: Prüfen, ob die Pipeline aktuell bereits läuft
    // ---------------------------------------------------------
    const workflowUrl = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowFileName}/runs?status=in_progress&branch=${pipelineBranch}`;
    const workflowCheck = await fetch(workflowUrl, { headers: githubHeaders });
    if (workflowCheck.ok) {
      const runData = await workflowCheck.json();
      if (runData.total_count > 0) {
        console.log("Pipeline läuft bereits. Dieser Durchlauf wird übersprungen.");
        return;
      }
    }

    // ---------------------------------------------------------
    // SCHRITT 2: Neuesten verfügbaren Lauf beim DWD ermitteln
    // ---------------------------------------------------------
    const utcNow = new Date(new Date().toUTCString());
    let dwdTargetTime = null;
    let dwdFolderString = "";

    for (let i = 0; i < 3; i++) {
      const checkTime = new Date(utcNow.getTime() - i * 60 * 60 * 1000);
      const YYYY = checkTime.getUTCFullYear();
      const MM = String(checkTime.getUTCMonth() + 1).padStart(2, '0');
      const DD = String(checkTime.getUTCDate()).padStart(2, '0');
      const HH = String(checkTime.getUTCHours()).padStart(2, '0');
      
      const folderStr = `${YYYY}-${MM}-${DD}T${HH}:00`;
      const dwdUrl = `https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/V_10M/r/${YYYY}-${MM}-${DD}T${HH}%3A00/s/PT024H00M.grib2`;
      
      const dwdResponse = await fetch(dwdUrl, { method: "HEAD" });
      if (dwdResponse.status === 200) {
        dwdTargetTime = checkTime;
        dwdFolderString = folderStr;
        console.log(`🎯 XNeuester verfügbarer DWD-Lauf gefunden: ${dwdFolderString}Z`);
        break; 
      }
    }

    if (!dwdTargetTime) {
      console.log("❌ Kein aktueller Datensatz auf dem DWD-Server im Abfragefenster gefunden.");
      return;
    }

    // ---------------------------------------------------------
    // SCHRITT 3: Über GitHub API prüfen, ob das finale webp existiert
    // ---------------------------------------------------------
    // Errechnet die Zielzeit des finalen PNGs (Startzeit + 24 Stunden)
    const pngTime = new Date(dwdTargetTime.getTime() + 24 * 60 * 60 * 1000);
    const pYYYY = pngTime.getUTCFullYear();
    const pMM = String(pngTime.getUTCMonth() + 1).padStart(2, '0');
    const pDD = String(pngTime.getUTCDate()).padStart(2, '0');
    const pHH = String(pngTime.getUTCHours()).padStart(2, '0');
    
    // Name der finalen Datei, z.B. "20260629_20Z.png"
    const pngFilename = `${pYYYY}${pMM}${pDD}_${pHH}Z.webp`; 
    
    // Pfad zur Datei im Repository (passe den Ordner an, falls die PNGs in einem Unterordner liegen)
    const githubFileUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${pngFilename}?ref=${dataBranch}`;

    console.log(`🔍 Prüfe GitHub-Repository auf existierendes PNG: ${pngFilename}`);
    
    const githubFileCheck = await fetch(githubFileUrl, {
      method: "HEAD", // HEAD reicht völlig aus, um die Existenz zu prüfen
      headers: githubHeaders
    });
    

    if (githubFileCheck.status === 200) {
      console.log(`💾 Datei ${pngFilename} existiert bereits im Repository. Lauf wird übersprungen.`);
      return;
    } else if (githubFileCheck.status === 404) {
      console.log(`🆕 Datei ${pngFilename} fehlt im Repository.`);
    } else {
      console.log(`⚠️ Unerwarteter Status von GitHub API: ${githubFileCheck.status}. Breche vorsichtshalber ab.`);
      return;
    }

// ---------------------------------------------------------
    // SCHRITT 4: GitHub Action gezielt für diesen Workflow & Branch triggern
    // ---------------------------------------------------------
    console.log(`🚀 Trigger die GitHub Action (${workflowFileName}) auf Branch '${pipelineBranch}' für Lauf ${dwdFolderString}Z...`);
    
    // Die korrekte URL steuert direkt die spezifische YAML-Datei an
    const dispatchUrl = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflowFileName}/dispatches`;
    
    const dispatchResponse = await fetch(dispatchUrl, {
      method: "POST",
      headers: githubHeaders,
      body: JSON.stringify({
        // Beim workflow_dispatch bestimmt das Feld 'ref' den Branch!
        ref: pipelineBranch, 
        // Die Parameter, die an das Python-Skript gehen
        inputs: {
          dwd_run: dwdFolderString
        }
      })
    });

    if (dispatchResponse.ok) {
      console.log("✅ GitHub Action erfolgreich gestartet.");
    } else {
      const errorText = await dispatchResponse.text();
      console.error(`❌ GitHub API Fehler (${dispatchResponse.status}):`, errorText);
    }  
  }
};