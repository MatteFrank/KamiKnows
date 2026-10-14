# Chat RAG da terminale

Lo script `scripts/chat_rag.py` carica i chunk e l'indice una volta. Ogni domanda
è indipendente: non usa la cronologia come contesto conversazionale. Non modifica
il benchmark e non assegna etichette di correttezza scientifica.

## Sessione interattiva GPU

Dal terminale del cluster richiedere un nodo GPU:

```bash
srun --account=kamiknows --qos=kamiknows --partition=bare-metal-GPU \
  --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash
```

Sul nodo assegnato, avviare Ollama nativo condiviso:

```bash
cd /home/e/eapicell/KamiKnows
export PATH="/storage-hpc/kamiknows/software/ollama/bin:$PATH"
export OLLAMA_MODELS=/storage-hpc/kamiknows/models/ollama
export OLLAMA_HOST=127.0.0.1:11448
ollama serve > /tmp/kamiknows-chat-ollama-${SLURM_JOB_ID}.log 2>&1 &
CHAT_OLLAMA_PID=$!
```

Quando il server è pronto, `ollama list` deve mostrare qwen3:0.6b. Per iniziare
con TF-IDF (senza caricamento di MiniLM):

```bash
/home/e/eapicell/.conda/envs/kamiknows/bin/python scripts/chat_rag.py \
  --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
  --retriever tfidf --source-policy explicit_arxiv \
  --model qwen3:0.6b --base-url http://127.0.0.1:11448
```

Per retrieval MiniLM sostituire `--retriever tfidf` con `--retriever minilm`.
MiniLM usa cache locale reale, CPU e la stessa politica di finestre di token dei
benchmark. Non scarica modelli né usa fallback hash. La query è il testo scritto
dall'utente: non aggiunge keyword/tipi di domanda del benchmark.

Esempi al prompt `KamiKnows>`:

```text
According to arXiv:1705.02355v2, which model is proposed?
Compare arXiv:1705.02355v2 and arXiv:2106.05285v3.
What are the limitations of fast calorimeter simulation?
/exit
```

Con ID espliciti cerca solo nei paper indicati; per più fonti riserva almeno un
passaggio per ciascuna. Senza ID cerca in tutto il corpus. Nomi come CaloGAN da
soli non sono vincoli. Fonti mancanti e budget insufficiente sono segnalati.

`--show-context` mostra i testi completi invece di preview; `--num-predict 1024`
aumenta il budget di generazione rispetto al default 512, se si desidera provare
questa variazione. Il testo appare a risposta completata, senza streaming token.
Warning di parsing e validazione citazioni vengono mostrati separatamente.

## Output e chiusura

Ogni sessione crea una directory nuova `outputs/chat_rag/<UUID>/` con:

- `session_manifest.json`: parametri, modello e hash del dataset;
- `contexts.jsonl`: contesto salvato prima di ogni chiamata;
- `answers.jsonl`: risposta, testo originale, citazioni, warning e stato backend,
  salvati e flushed appena il turno termina.

`--session-dir outputs/chat_rag/prova_v1` sceglie un nome; directory già esistenti
sono rifiutate. `/exit`, EOF o Ctrl+C chiudono la chat; i turni completati restano
salvati. Il server Ollama avviato separatamente resta attivo finché lo si termina.

Alla fine, nel medesimo terminale:

```bash
kill "$CHAT_OLLAMA_PID"
exit
```

L'ultimo `exit` rilascia la shell interattiva Slurm. Risposte con citazioni valide
possono comunque essere errate: la revisione scientifica resta necessaria.
