# V3: ispezione preliminare delle quattro astensioni su domande rispondibili

## Obiettivo, input e metodo

Confrontare il testo originale di Qwen 4B, i cinque chunk forniti e le bozze di
riferimento per pair_01_implicit, pair_06_explicit, pair_07_explicit e
pair_07_implicit. Input: JSONL della run
`outputs/qa_grounded_qwen4b_minilm_eval_v3_paired_top5/` e
`data/eval/rag_eval_set_v3_reference_answers_draft.jsonl`.
Ispezione locale dei testi, senza nuovi modelli o modifiche alla pipeline.
Le osservazioni sono dell'assistente e non etichette validate da un revisore umano.
I campi reviewer del CSV esistente rimangono intatti e pending.

## Provenienza del campo limitations

Nelle quattro risposte ispezionate limitations è presente nel JSON originale
restituito da Qwen. Non è una spiegazione scritta dall'assistente e non prova
oggettivamente che l'evidenza manchi. Il prompt richiede quel campo; il parser
lo conserva/normalizza. Nei tre record di questa run con json_parse_error, invece,
la pipeline inserisce automaticamente la frase tecnica
`Response was not valid JSON; parsed conservatively.`:
pair_03_implicit, pair_06_implicit e pair_08_explicit.
Le bozze reference_answer_draft e i criteri di valutazione sono stati preparati
dall'assistente dai testi locali e restano da validare.

## Pair 01, implicita: CaloGAN

Qwen dichiara che non è descritto un simulatore con le caratteristiche richieste.
Eppure C1 (1705.02355v2, Experimental Results) nomina CaloGAN e dice che i dati
provengono da un calorimetro LAr a tre strati con segmentazione eterogenea.
Il titolo del paper nel contesto descrive anche la famiglia GAN; C2 si riferisce
ai campioni GAN. La motivazione non riconosce informazioni già visibili.

Diagnosi preliminare: problema di lettura/collegamento del contesto da parte del
generatore, con distrattori da altri paper; non semplice assenza del paper giusto.
Il chunk Method sarebbe un supporto più diretto per la famiglia del modello.

## Pair 06, esplicita: CaloGAN contro CaloFlow e likelihood

Le due fonti richieste sono presenti, ma quattro chunk sono di CaloFlow e quello
di CaloGAN (C5) riguarda la classificazione, non la formulazione generativa.
Non viene recuperato il passaggio Method di CaloGAN che nega esplicitamente
l'accesso a una likelihood. D'altra parte C4 (CaloFlow, Conclusions) afferma
esplicitamente i vantaggi delle normalizing flows: likelihood trattabili,
training più stabile e selezione tramite negative log-likelihood.

Diagnosi preliminare: retrieval di passaggi non abbastanza diretti per entrambi
i lati del confronto, più una limitations troppo ampia che nega anche evidenza
effettivamente presente. L'astensione sulla risposta completa è più comprensibile
che nella coppia 01; una risposta parziale con limiti espliciti sarebbe possibile.

## Pair 07, esplicita: jet contro shower elettromagnetici

C5 (studio jet, Introduction) descrive chiaramente la generazione della risposta
ricostruita condizionata al contenuto particle-level e le immagini gray-scale con
intensità legata all'energia. C3/C4 descrivono le deposizioni nei layer e gli
shower elettromagnetici del WGAN. Non arrivano i migliori passaggi di geometria:
non sono espliciti tutti i numeri della bozza (32x32, sette strati).

Qwen dice che manca un confronto già scritto side-by-side. Un confronto richiesto
dall'utente può però essere costruito da evidenze separate: non deve già apparire
nel testo di un paper. La domanda chiede oggetti/rappresentazioni, non quei numeri
esatti, quindi una risposta qualitativa è supportabile senza inventare geometrie.

Diagnosi preliminare: difficoltà di sintesi del generatore e contesto migliorabile;
non totale mancanza di informazioni su entrambi gli studi.

## Pair 07, implicita

C2 è lo stesso passaggio utile dello studio jet. C3 e C5 del WGAN descrivono
layer calorimetrici e shower elettromagnetici isolati. Ancora una volta la
limitations richiede un confronto diretto già presente, invece di combinare i
fatti supportati dalle due fonti. Anche qui mancano le geometrie dettagliate della
bozza ma non gli elementi per un confronto qualitativo.

Diagnosi preliminare: simile alla variante esplicita. Non basta attribuire
l'astensione all'assenza di un vincolo sulle fonti: entrambi i paper sono presenti.

## Verifica, limiti e prossimo passo

Confrontati raw_model_response e campi normalizzati; tutte e quattro le risposte
sono JSON valido con answer vuoto e insufficient_evidence. Non sono astensioni
introdotte dal fallback del parser. Nessun test software necessario: artefatto
di sola analisi; nessuna modifica al CSV di revisione o ai risultati esistenti.

Le limitations sono auto-spiegazioni del modello, non diagnosi attendibili di per
sé. Prima di cambiare il retrieval, un test controllato utile è ripetere questi
quattro casi con passaggi di riferimento selezionati e parametri invariati, poi
confrontare il prompt originale con un'istruzione esplicita per sintesi tra fonti
e risposte parziali supportate. Questi test non sono stati eseguiti qui.
