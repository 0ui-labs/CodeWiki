I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.

## Beobachtungen

Die `pyproject.toml` enthält bereits `networkx>=3.5` (Zeile 48) und alle LLM-Provider-SDKs (anthropic, google-genai, groq, cerebras-cloud-sdk, tiktoken). **Jedoch fehlen zwei kritische Dependencies für Plan 11 (Watcher)**: `mcp>=1.0.0` (sollte aus Plan 09 vorhanden sein, ist aber nicht da) und `watchdog>=4.0.0` (für Filesystem-Monitoring). Die Dependencies sind alphabetisch/thematisch gruppiert (CLI-Tools, Tree-Sitter, LLMs, Utilities).

## Ansatz

Ergänze die beiden fehlenden Dependencies in der `dependencies`-Liste unter `[project]`: `mcp>=1.0.0` (für FastMCP-Server aus Plan 09) nach den Mermaid-Paketen und `watchdog>=4.0.0` (für Filesystem-Observer) nach `tiktoken`. Alphabetische Sortierung wird nicht strikt eingehalten (bestehende Struktur priorisiert thematische Gruppierung), daher platziere ich sie logisch: MCP bei Web/Server-Tools, Watchdog bei Utilities. Teste Installation und CLI-Verfügbarkeit.

## Implementierungsschritte

### 1. Dependencies in pyproject.toml ergänzen

**Datei:** `pyproject.toml`

- **Zeile 52 (nach `mermaid-py>=0.8.0`)**: Füge `"mcp>=1.0.0",` hinzu
- **Zeile 58 (nach `tiktoken>=0.7.0`)**: Füge `"watchdog>=4.0.0"` hinzu (ohne Komma, da letzter Eintrag)

**Begründung der Platzierung:**
- `mcp` gehört zu Server/Protokoll-Dependencies (nahe bei Mermaid/Web-Tools)
- `watchdog` ist ein Utility (wie psutil, PyYAML) → am Ende der Liste

**Resultierende Struktur (Zeilen 26-59):**
```toml
dependencies = [
    # ... (bestehende CLI/Tree-Sitter/LLM deps) ...
    "mermaid-parser-py>=0.0.2",
    "mermaid-py>=0.8.0",
    "mcp>=1.0.0",                    # NEU
    "anthropic>=0.67.0",
    "google-genai>=1.36.0",
    "groq>=0.30.0",
    "cerebras-cloud-sdk>=1.0.0",
    "tiktoken>=0.7.0",
    "watchdog>=4.0.0"                # NEU
]
```

### 2. Installation testen

**Terminal-Befehle:**

```bash
# Im Repo-Root (/Users/philippbriese/.../CodeWiki/)
pip install -e .
```

**Erwartetes Verhalten:**
- Erfolgreiche Installation von `mcp` und `watchdog` (plus deren Transitive Dependencies)
- Keine Konflikte mit bestehenden Packages
- Output zeigt "Successfully installed mcp-X.X.X watchdog-X.X.X ..."

### 3. CLI-Verfügbarkeit verifizieren

**Terminal-Befehl:**

```bash
codewiki serve --help
```

**Erwartetes Verhalten:**
- Zeigt Help-Text für `serve`-Command (aus Plan 09/12)
- Keine Import-Errors (insbesondere keine `ModuleNotFoundError: No module named 'mcp'` oder `'watchdog'`)
- Output enthält `--docs-dir` Option und Beschreibung

**Fallback bei Fehler:**
- Falls `codewiki serve --help` fehlschlägt: Prüfe `codewiki --help` (listet alle Commands)
- Falls MCP-Import fehlt: Verifiziere `python -c "import mcp; print(mcp.__version__)"`
- Falls Watchdog-Import fehlt: Verifiziere `python -c "import watchdog; print(watchdog.__version__)"`

### 4. Optionale Verifikation (für Vollständigkeit)

**Prüfe alle kritischen Imports:**

```bash
python -c "
import mcp
import watchdog
import networkx
print('✓ All Plan 11 dependencies available')
"
```

**Erwartetes Verhalten:**
- Kein Output außer "✓ All Plan 11 dependencies available"
- Exit-Code 0

---

## Zusammenfassung

| Dependency | Version | Status | Zweck |
|------------|---------|--------|-------|
| `networkx` | `>=3.5` | ✓ Vorhanden | Graph-Operationen (DependencyGraphService) |
| `mcp` | `>=1.0.0` | ✗ Fehlt → **Hinzufügen** | FastMCP Server (Plan 09) |
| `watchdog` | `>=4.0.0` | ✗ Fehlt → **Hinzufügen** | Filesystem-Monitoring (Plan 11) |

**Änderungen:** 2 Zeilen in `pyproject.toml` (Zeilen 54 und 59)  
**Tests:** Installation + CLI-Help (< 2 Minuten)  
**Risiko:** Minimal (keine Breaking Changes, nur Additions)