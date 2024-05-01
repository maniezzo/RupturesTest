# Raptures
## Metodi di ricerca
### Pelt
Pelt è usato per stimare il numero e la posizione dei punti di cambiamento in una serie temporale. Questo algoritmo penalizza l'aggiunta di ogni punto di cambiamento con un valore di penalità, cercando la segmentazione con il costo più basso, considerando sia il costo della segmentazione che la penalità. 

L'idea di base è che un punto di cambiamento deve ridurre il costo della segmentazione di più della penalità per essere considerato valido. 
La scelta della penalità è critica: penalità basse possono causare il rilevamento di falsi punti di cambiamento, mentre penalità alte possono far perdere dei punti di cambiamento reali.

Parametri:

![img_1.png](img/img_1.png)

Predict parametri:

![img_4.png](img/img_4.png)

### BinarySeg

La segmentazione binaria è piuttosto semplice: innanzitutto cerca un singolo punto di cambiamento nell'intero segnale.

Una volta trovato quel punto, divide il segnale in due parti e ripete il processo per ciascuna di quelle parti.

Ciò continua finché non vengono trovati più punti di modifica o non viene soddisfatto un criterio di arresto specificato.

Ha una bassa complessità, il che significa che non richiede troppo tempo o potenza di calcolo per essere eseguito ed è una buona opzione per set di dati di grandi dimensioni.

Uno svantaggio è che a volte può perdere punti di cambiamento o rilevarne di falsi, soprattutto quando i cambiamenti sono ravvicinati o il segnale è rumoroso.

Inoltre prende decisioni basate sulla migliore scelta immediata senza considerare l'impatto complessivo sul risultato finale.


Parametri:

![img_2.png](img/img_2.png)

Predict parametri:

![img_5.png](img/img_5.png)

### Dynp
DynP sfrutta un approccio di programmazione dinamico per ordinare in modo efficiente la ricerca su tutte le possibili segmentazioni, il che aiuta a ottimizzare il processo e fornire risultati accurati.

Funziona esaminando sistematicamente tutte le possibili segmentazioni di un dato segnale per trovare il minimo esatto della somma dei costi associati a ciascuna segmentazione.

Tuttavia, un requisito importante per l'utilizzo di DynP è che l'utente debba specificare in anticipo il numero di punti di modifica.

Anche se in alcuni casi questo potrebbe rappresentare un limite, la flessibilità e la precisione del metodo lo rendono uno strumento prezioso per il rilevamento dei punti di cambiamento quando il numero di punti di cambiamento è noto o può essere stimato.

La complessità computazionale di DynP può essere un altro fattore limitante, soprattutto quando si ha a che fare con set di dati di grandi dimensioni o funzioni di costo più complesse.

L'algoritmo potrebbe diventare lento o addirittura poco pratico per alcune applicazioni a causa del suo elevato costo computazionale.

Se non conosci in anticipo il numero di punti di modifica, puoi considerarlo un altro iperparametro da ottimizzare.

Ho deciso di eseguire un ciclo per provare valori diversi per il numero di punti di modifica e ispezionare visivamente i risultati per vedere quale aveva più senso.

Parametri:


![img.png](img/img.png)

Predict parametri:

![img_3.png](img/img_3.png)

## Spiegazione Parametri
### Model
Specifica il modello di rilevamento dei cambiamenti utilizzato.

Esistono diversi tipi di modelli:

+ ***l1*** (Least Absolute Deviation): Questo metodo minimizza la somma degli scarti assoluti tra i punti dati e la linea di regressione. È utile quando si desidera una robustezza maggiore agli outlier rispetto al metodo dei minimi quadrati.
+ ***l2*** (Least Squared deviation): Questo metodo minimizza la somma degli scarti quadrati tra i punti dati e la linea di regressione. È sensibile agli outlier, ma può essere influenzato in modo significativo da essi.
+ ***rbf*** (Kernelized mean change): Questo metodo utilizza una funzione kernel per calcolare la similarità tra i segmenti di dati. È particolarmente utile quando i cambiamenti non sono necessariamente lineari e quando è importante catturare i cambiamenti non solo in valore ma anche in forma.
+ ***normal*** (Gaussian process change): Questo metodo utilizza un processo gaussiano per modellare i dati e individuare i cambiamenti. È utile quando si desidera modellare la correlazione tra i dati e avere una stima della distribuzione dei cambiamenti.
+ ***cosine*** (CostCosine): Questo metodo calcola il costo del cambiamento utilizzando la similarità coseno tra i segmenti di dati. Può essere utile quando si desidera rilevare cambiamenti basati sulla direzione dei vettori dei dati.
+ ***linear*** (Linear model change): Questo metodo assume che i dati seguano un modello lineare e cerca i punti in cui il modello cambia. È utile quando si sospetta che i cambiamenti seguano un modello lineare.
+ ***clinear*** (Continuous linear change): Questo metodo assume che i cambiamenti seguano un modello lineare ma possono essere continui nel tempo. È utile quando si desidera rilevare cambiamenti graduati e non repentini.
+ ***rank*** (Rank-based change): Questo metodo si basa sulle classifiche dei dati anziché sui valori effettivi. È utile quando si desidera rilevare cambiamenti nella distribuzione dei dati piuttosto che nei valori stessi.
+ ***mahalanobis*** (Mahalanobis-type change): Questo metodo utilizza la distanza di Mahalanobis per valutare i cambiamenti nei dati. È particolarmente utile quando i dati hanno correlazioni e varianze diverse.
+ ***ar*** (Autoregressive model change): Questo metodo utilizza modelli autoregressivi per rilevare cambiamenti nei dati nel tempo. È utile quando i dati hanno una struttura temporale e si desidera modellare le dipendenze temporali per rilevare i cambiamenti.
+ ***costum*** (Custom): il modello costum consente agli utenti di definire manualmente una funzione di costo personalizzata per identificare i cambiamenti nei dati. Offre una maggiore flessibilità nella definizione dei criteri di cambiamento.

## Custom Cost

La funzione error() nella classe MyCost è il cuore del calcolo del costo personalizzato per la rilevazione dei punti di rottura. Vediamo i dettagli della sua implementazione:

```
class MyCost(BaseCost):
    """Custom cost for percentage difference between segment medians."""
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end]."""
        segment = self.signal[start:end]
        segment_mean = np.mean(segment)
        absolute_diff = np.abs(segment - segment_mean)
        cost = np.sum(absolute_diff)
        return cost
```


La funzione error() accetta due argomenti, start e end, che indicano gli indici di inizio e fine del segmento di interesse all'interno del segnale.

Calcolo del segmento: Viene estratto il segmento di segnale specificato dagli indici start e end. Questo segmento rappresenta la porzione del segnale su cui calcoleremo il costo del punto di rottura.

```
segment = self.signal[start:end]
```

Calcolo della media del segmento: Viene calcolata la media del segmento. Questo ci fornisce un valore rappresentativo per il segmento.

```
segment_mean = np.mean(segment)
```

Calcolo della differenza assoluta rispetto alla media: Viene calcolata la differenza assoluta tra ciascun punto del segmento e la media del segmento. Questo ci fornisce una misura di quanto ciascun punto si discosti dalla media del segmento.

```
absolute_diff = np.abs(segment - segment_mean)
```

Calcolo del costo totale: La somma delle differenze assolute calcolate nel passaggio precedente rappresenta il costo del segmento. Questo valore può essere considerato come una misura di quanto il segmento differisca dalla sua media.

```
cost = np.sum(absolute_diff)
```

Restituzione del costo: Il costo calcolato viene restituito come risultato della funzione error(). Questo valore sarà utilizzato dall'algoritmo di rilevamento dei punti di rottura per valutare la bontà della segmentazione in corrispondenza del punto di rottura specificato.

```
return cost
```

In sintesi, la funzione error() calcola il costo di un segmento di segnale specifico, che è la somma delle differenze assolute tra ciascun punto del segmento e la media del segmento stesso. Questo costo personalizzato viene quindi utilizzato per guidare l'algoritmo di rilevamento dei punti di rottura nella ricerca dei punti di rottura ottimali nella serie temporale.

## Annotazioni

### BTCEuro

In base al dataset utilizzato, secondo me è meglio utilizzare come modello sia per BinarySeng
che per Dynp ***l1***. L'obbietto della nostra analisi è quello di monitorare
l'anadmaento del Bitcoin rispetto all'euro. Il Bitcoin è noto per la sua volatilità, che può
causare outlier (dati anomali che si discosatno totalmente rispetto al dataset, dovuti a fattori esterni).
L1 è un modello noto per la sua robustezza agli outlier, infitta minimizza la somma degli scarti assoluti tra 
i punti dati e linea di regresseione, consentendo di effettuare delle analisi più accurate riducendo l'influenza di fattori esterni.


## TODO
- capire iperparametri e vedere se ce ne sono altri utilizzabili
