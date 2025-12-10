I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.

## Beobachtungen

Die `pyproject.toml` enthält bereits umfangreiche Dependencies für LLM-Provider (Anthropic, OpenAI, Google, Groq, Cerebras), Tree-Sitter-Parser, und Utilities wie NetworkX, Rich, und Pydantic. Die Dependencies sind thematisch gruppiert (Parser, LLM-SDKs, Utilities). Es fehlen noch die für Plan 12 (Semantic Navigation/RAG) benötigten Pakete: LanceDB (serverlose Vektor-DB), Sentence-Transformers (lokale Embeddings), und PyTorch (ML-Framework). Keine offensichtlichen Konflikte erkennbar - die neuen Pakete sind unabhängig von bestehenden Dependencies.

## Ansatz

Füge die drei neuen Dependencies am Ende der bestehenden Liste hinzu (nach `tiktoken`), gruppiert als "Semantic Search / RAG" Block. Dies folgt der bestehenden Struktur und macht die Zugehörigkeit klar. Die Versionsangaben (`>=`) erlauben Flexibilität für Patch-Updates, während Mindestversionen Kompatibilität sicherstellen:
- `lancedb>=0.5.0`: Stabile API seit 0.5.x, serverless embedded mode
- `sentence-transformers>=2.5.0`: Unterstützt moderne Modelle wie `all-MiniLM-L6-v2`
- `torch>=2.0.0`: Kompatibel mit sentence-transformers 2.5+, nutzt moderne PyTorch-Features

Keine Änderungen an bestehenden Dependencies nötig. Verifizierung via `pip install -e .` (prüft Resolver-Konflikte) und `codewiki serve --help` (stellt sicher, dass CLI funktioniert und keine Import-Fehler auftreten).

## Implementierungsschritte

### 1. Ergänze Dependencies in pyproject.toml

**Datei:** `pyproject.toml`

**Änderungen:**
- Füge nach Zeile 57 (`tiktoken>=0.7.0"`) drei neue Zeilen hinzu:
  ```toml
  "mcp>=1.0.0",
  "watchdog>=4.0.0",
  "lancedb>=0.5.0",
  "sentence-transformers>=2.5.0",
  "torch>=2.0.0"
  ```

**Begründung:**
- `mcp` und `watchdog` wurden in vorherigen Plänen bereits erwähnt, sind aber noch nicht in der Datei (basierend auf Plan 12/18/20)
- Die neuen RAG-Dependencies werden als Block am Ende platziert für klare Zuordnung
- Reihenfolge: MCP/Watcher (Infrastruktur) → RAG-Stack (LanceDB → Transformers → PyTorch als Foundation)

**Resultierende Struktur (Zeilen 26-62):**
```toml
dependencies = [
    "click>=8.1.0",
    "keyring>=24.0.0",
    "GitPython>=3.1.40",
    "Jinja2>=3.1.6",
    "tree-sitter>=0.23.2",
    "tree-sitter-language-pack>=0.8.0",
    "tree-sitter-python>=0.23.6",
    "tree-sitter-java>=0.23.5",
    "tree-sitter-javascript>=0.21.4",
    "tree-sitter-typescript>=0.21.2",
    "tree-sitter-c>=0.21.4",
    "tree-sitter-cpp>=0.23.4",
    "tree-sitter-c-sharp>=0.23.1",
    "openai>=1.107.0",
    "litellm>=1.77.0",
    "pydantic>=2.11.7",
    "pydantic-settings>=2.10.1",
    "pydantic-ai>=1.0.6",
    "requests>=2.32.4",
    "python-dotenv>=1.1.1",
    "rich>=14.1.0",
    "networkx>=3.5",
    "psutil>=7.0.0",
    "PyYAML>=6.0.2",
    "mermaid-parser-py>=0.0.2",
    "mermaid-py>=0.8.0",
    "anthropic>=0.67.0",
    "google-genai>=1.36.0",
    "groq>=0.30.0",
    "cerebras-cloud-sdk>=1.0.0",
    "tiktoken>=0.7.0",
    "mcp>=1.0.0",
    "watchdog>=4.0.0",
    "lancedb>=0.5.0",
    "sentence-transformers>=2.5.0",
    "torch>=2.0.0"
]
```

### 2. Verifiziere Installation und Kompatibilität

**Kommandos:**
```bash
# Installiere Paket im Entwicklungsmodus
pip install -e .

# Prüfe CLI-Verfügbarkeit (stellt sicher, keine Import-Fehler)
codewiki serve --help
```

**Erwartete Ausgabe:**
- `pip install -e .`: Erfolgreiche Installation aller Dependencies ohne Resolver-Konflikte
- `codewiki serve --help`: Zeigt Help-Text für `serve` Command ohne Fehler

**Fallback bei Konflikten:**
Falls pip Konflikte meldet (unwahrscheinlich):
1. Prüfe Konflikt-Details in pip-Output
2. Passe Versionsangaben an (z.B. `torch>=2.0.0,<2.5.0` falls neuere Versionen inkompatibel)
3. Dokumentiere Änderungen

### 3. Optionale Verifikation: Import-Test

**Kommando:**
```bash
python -c "import lancedb; import sentence_transformers; import torch; print('✓ All imports successful')"
```

**Erwartete Ausgabe:**
```
✓ All imports successful
```

**Hinweis:** Beim ersten Import von `sentence-transformers` kann ein Download des Modells (~80MB) erfolgen - dies ist normal und wird in Plan 12 (EmbeddingService) behandelt.

### 4. Dokumentation (Optional)

Falls ein `requirements.txt` existiert (für Docker/CI), synchronisiere es:
```bash
pip freeze > requirements.txt
```

**Hinweis:** Dies ist optional, da `pyproject.toml` die primäre Quelle ist.

## Zusammenfassung

| Dependency | Version | Zweck | Größe (ca.) |
|------------|---------|-------|-------------|
| `lancedb` | >=0.5.0 | Serverlose Vektor-DB für Embeddings | ~10 MB |
| `sentence-transformers` | >=2.5.0 | Lokale Embedding-Modelle (Privacy-First) | ~50 MB + Modelle |
| `torch` | >=2.0.0 | ML-Framework (Dependency von sentence-transformers) | ~800 MB (CPU) |

**Gesamtgröße:** ~860 MB (einmalig beim ersten Install)

**Keine Breaking Changes:** Alle bestehenden Funktionen bleiben unverändert. Die neuen Dependencies werden erst in nachfolgenden Phasen (Plan 12+) aktiv genutzt.