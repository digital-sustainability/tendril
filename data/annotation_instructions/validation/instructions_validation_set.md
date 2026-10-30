Jede Aufgabe zeigt ein **Handlungsfeld**, ein **Nachhaltigkeitskriterium** (beides grün) und eine **gelb markierte Textstelle** aus einem Ausschreibungsdokument. Der graue Text davor und danach ist nur Kontext und wird nicht bewertet.

Es werden **zwei getrennte Bewertungen** abgegeben: einmal für das **Kriterium** und einmal für das **Handlungsfeld**.

### **Die zwei Fragen**

Das **Handlungsfeld** ist das übergeordnete Ziel. Das **Kriterium** ist eine konkrete Umsetzung dieses Ziels. Das Kriterium ist also enger als das Handlungsfeld.

| **Handlungsfeld** | Lebensmittelabfälle vermeiden. Möglichst wenige Lebensmittelabfälle. |
| ----------------- | -------------------------------------------------------------------- |
| **Kriterium**     | Keine Teller werden zu Demonstrationszwecken angerichtet.            |

Eine Textstelle über Massnahmen gegen Foodwaste allgemein ist **relevant für das Handlungsfeld**, aber **nicht relevant für dieses konkrete Kriterium**. Wichtig: wir annotieren Handlungsfelder nur dann als (teilweise) relevant, wenn **ein Nachhaltigkeitsbezug besteht**, auch wenn ein Handlungsfeld nur sehr wenig Text enthält.

**Wichtige Konsistenzregel:** Weil das Kriterium im Handlungsfeld enthalten ist, gilt in der Regel: Wer beim Kriterium 2 vergibt, sollte beim Handlungsfeld mindestens ebenso hoch bewerten. Die Bewertung fürs Handlungsfeld ist also **meistens nicht tiefer** als die fürs Kriterium. Ausnahmen können für vereinzelte Handlungsfelder auftreten, die sehr kurz oder allgemein gehalten sind und dementsprechend nicht unbedingt alle Aspekte des spezifischeren Kriteriums abdecken.

_Der umgekehrte Fall ist dagegen häufig und erwünscht: Handlungsfeld 2, Kriterium 0. Genau diese Fälle sind der Grund für die zweite Frage._

### **Entscheidungsweg (für beide Fragen gleich)**

1. Ist die Textstelle überhaupt **lesbar**?  
   → Wenn nein: Nicht auswertbar (bei **beiden** Fragen)
2. Behandelt sie das **gleiche sachliche Thema**?  
   → Wenn nein: 0
3. Liefert sie **echte Evidenz** statt Boilerplate oder Formatierungsrauschen?  
   → Wenn nein: 0
4. Sind die Hinweise **indirekt, unvollständig oder mehrdeutig**?  
   → Wenn ja: 1
5. Sind die Hinweise **explizit, konkret und klar ausgerichtet**?  
   → Wenn ja: 2

_Schritt 2 bis 5 werden zweimal durchlaufen: einmal gegen den Kriteriumstext, einmal gegen das Handlungsfeld. Schritt 1 gilt für beide gemeinsam._

_Es gibt kein Label für „unsicher". Mehrdeutige oder nur mit Kontext verständliche Evidenz ist genau der Fall für 1 (Schritt 4)._

### **Wichtigste Abgrenzung: 0 oder Nicht auswertbar?**

**→ 0** wenn man erkennt, **was der Text aussagt**, er aber nicht passt.

**→ Nicht auswertbar** wenn man **nicht erkennt, was der Text überhaupt aussagt**.

_Also kein Label für inhaltlich wertlosen, aber lesbaren Text. Verwaltungstext, Boilerplate, leere Tabellen mit lesbaren Überschriften und wiederholte URLs sind lesbar und erhalten 0._

**Bei beiden Fragen gleich setzen:** Unlesbarkeit ist eine Eigenschaft des Textes, nicht der Frage. Wenn Nicht auswertbar, dann bei Kriterium **und** Handlungsfeld.

### **Tabellen sind reguläre Annotationseinheiten**

Eine Textstelle ist nicht immer ein grammatikalischer Satz. Fliesstext wird in Sätze segmentiert, Tabellen aus Word und Excel bleiben als ganze Tabelle erhalten.

**Entscheidend ist der Inhalt, nicht die Form.** Eine knappe, aber verständliche Tabelle wird normal mit 0/1/2 bewertet. Sie ist kein Fehler und nicht automatisch 0.

### **Label-Definitionen**

_Die folgenden Definitionen gelten für beide Fragen. „Bezugstext" meint je nach Frage den Kriteriumstext oder das Handlungsfeld._

**0 — Nicht relevant**

Die Textstelle ist **lesbar**, liefert aber **keine aussagekräftigen Hinweise** auf den Bezugstext.

- Anderes Thema als der Bezugstext
- Rein administrativ, verfahrenstechnisch, rechtlich oder formatierungsbezogen
- Generische Projektboilerplate ohne Nachhaltigkeitsinhalt
- Nur lexikalische Überlappung, keine inhaltliche Übereinstimmung
- Lesbare Tabellenüberschrift, Listenlabel, URL-Block oder Dateiname ohne Aussage

_Beispiele: Einreichungsfristen, Kontaktdaten, Verweise auf Anlagen; generische Bewertungsformulare; Tabellen ohne Anforderung, Verpflichtung oder Nachweis; eine Tabelle, in der dieselbe URL in jeder Zelle wiederholt wird._

**1 — Teilweise relevant**

Die Textstelle ist **mit dem Bezugstext verknüpft**, die Hinweise sind aber **unvollständig, indirekt, schwach, mehrdeutig oder nur teilweise ausgerichtet**.

- Richtiges Thema, aber nicht die vollständige Anforderung
- Einhaltung wird angedeutet, aber nicht explizit gemacht
- Nur ein Teilaspekt eines mehrteiligen Bezugstexts
- Wird erst im Zusammenspiel mit dem Kontext sinnvoll
- Schwächeres Ambitionsniveau als gefordert

_Beispiele: Klimaziele ohne messbare Zielwerte oder Verifikation; Energiemanagement ohne konkrete Massnahmen; allgemeine Nachhaltigkeitsaussagen ohne operative Details; relevantes Tabellenfragment mit zu wenig Substanz._

**2 — Klar relevant**

Die Textstelle liefert **klare, direkte, passende Hinweise** zum Bezugstext.

- Formuliert explizit Anforderung, Verpflichtung, Qualifikation, Massnahme oder Nachweis
- Konkrete Nachhaltigkeitsinhalte in direkter Übereinstimmung mit dem Bezugstext
- Spezifische Schwellenwerte, Standards, Zertifizierungen, Prozesse oder Pflichten
- Auch eine Tabellenzelle mit konkreter Anforderung kann 2 erhalten, z.B. „CO₂-Emissionen: < 50 g/km"

_Beispiele: FSC/PEFC-zertifiziertes Holz bei einem Holz-Bezugstext; konkrete Emissionsziele inkl. Messung und Validierung; klare betriebliche Anforderungen zu Energieeinsparung oder Kühleffizienz._

**Nicht auswertbar — Extraktionsfehler**

Die Textstelle ist **durch einen Fehler der Dokumentextraktion nicht lesbar** und lässt keine inhaltliche Bewertung zu. Gilt immer für **beide** Fragen gleichzeitig.

- Wörter mitten im Wort zerrissen oder Wortanfänge fehlen
- Leere Formularfelder, die nur aus Punktlinien oder Unterstrichen bestehen
- Zeichensalat, Encoding-Fehler, unlesbar ineinandergeschobene Fragmente

**Nicht auswertbar:**

```
gistikkonzept und des L konzeptes si Baustelle llation edarfs (Hebe bungsphase ogistikhandb g der Baus
für das Kommunalfahrzeug: ……………………………………....................
```

**Dagegen normal bewerten (0/1/2):**

Knappe, aber verständliche Tabellen wie Typ | Bedarf virtual Cores | Anzahl; Stichworte und Aufzählungen ohne vollständige Sätze; Fachsprache, Abkürzungen oder Normverweise, die verständlich sind.

_Diese Fälle sollten selten sein. Die Vorverarbeitung filtert einen Grossteil solcher Artefakte bereits automatisch heraus. Wenn Nicht auswertbar häufig nötig scheint, bitte melden, statt das Label grosszügig zu vergeben._

### **Typische Kombinationen**

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Kriterium</strong></p></th><th><p><strong>Handlungsfeld</strong></p></th><th><p><strong>Bedeutung</strong></p></th></tr><tr><td><pre><code>2</code></pre></td><td><pre><code>2</code></pre></td><td><p>Textstelle trifft die konkrete Anforderung. Normalfall bei starker Evidenz.</p></td></tr><tr><td><pre><code>0</code></pre></td><td><pre><code>2</code></pre></td><td><p>Textstelle passt zum übergeordneten Ziel, aber nicht zur konkreten Anforderung.&nbsp;<strong>Häufig und erwünscht.</strong></p></td></tr><tr><td><pre><code>0</code></pre></td><td><pre><code>1</code></pre></td><td><p>Thematisch im Umfeld des Ziels, aber schwach oder indirekt.</p></td></tr><tr><td><pre><code>0</code></pre></td><td><pre><code>0</code></pre></td><td><p>Textstelle passt zu keinem von beidem. Häufigster Fall.</p></td></tr><tr><td><pre><code>2</code></pre></td><td><pre><code>0</code></pre></td><td><p><strong>Sollte nie oder nur selten auftreten</strong>&nbsp;— bitte Bewertung prüfen.</p></td></tr></tbody></table></div>

### **Prinzipien**

#### **Textstelle vs. Kontext**

Der Kontext darf zur Orientierung genutzt werden, bewertet wird aber ausschliesslich die gelb markierte Textstelle. Gilt für beide Fragen.

- Nur zusammen mit Kontext klar relevant: **maximal 1**
- 2 erfordert, dass die Textstelle eigenständig klare Evidenz liefert

_Hintergrund: Im späteren Benchmark wird die Textstelle ohne ihren Kontext bewertet. Ein Label, das sich nur aus dem Kontext ergibt, wäre für ein Modell nicht nachvollziehbar._

#### **Ambitionsniveaus**

Sie dienen als Orientierung, wie stärkere Evidenz aussehen kann, und helfen bei der Abgrenzung zwischen schwächerer und stärkerer Erfüllung. Sie beziehen sich auf das **Kriterium**, nicht auf das Handlungsfeld. Bei einem Grossteil der Kriterien sind die drei Niveaus identisch; dann erscheint nur eine Zeile mit dem Hinweis „Gleich für alle drei Niveaus". Das ist korrekt und bedeutet lediglich, dass das Kriterium nicht nach Ambitionsniveau differenziert.

#### **Notizfeld**

Optional für Auffälligkeiten: unklare Kriteriumsformulierungen, systematische Grenzfälle, wiederkehrende Extraktionsprobleme. Notizen fliessen nicht in das Label ein, helfen aber bei der Verfeinerung der Richtlinien.