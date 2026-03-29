# Anthropomorphic Finger: Largest Ellipse (Actor-Critic with Antagonist)

Der Fokus liegt auf einem kontinuierlichen Actor-Critic-Algorithmus für das Environment `FingerEllipseEnv`, in dem der Finger eine große, nicht-degenerierte Ellipsen-Trajektorie lernen soll.

---
## Overview

In dieser Arbeit wird ein Reinforcement-Learning-Ansatz zur Steuerung eines anthropomorphen Fingers mit drei Freiheitsgraden untersucht. Der Finger wird als planar aufgebauter Roboterarm mit drei rotatorischen Gelenken und festen Linklängen modelliert. Ziel des Lernprozesses ist es, eine Trajektorie der Fingerspitze zu erzeugen, die eine möglichst große Ellipse im erreichbaren Arbeitsraum einschließt.

Zur Lösung dieses Problems wurde ein eigenes Simulations-Environment implementiert, in dem der Agent kontinuierliche Aktionen in Form von Gelenkwinkeländerungen ausführt. Der Zustand des Systems umfasst die Gelenkwinkelrepräsentation, die aktuelle Position der Fingerspitze sowie eine normierte Zeitphase der Episode. Die Qualität einer erzeugten Trajektorie wird über eine Ellipsenschätzung auf Basis der Kovarianz der Trajektorienpunkte bestimmt, wodurch sich Fläche, Achsenverhältnis und weitere geometrische Eigenschaften ableiten lassen.

Als Lernverfahren wird ein Actor–Critic-Algorithmus mit kontinuierlicher Policy eingesetzt. Der Actor erzeugt eine stochastische Aktionsverteilung über die Gelenkbewegungen, während der Critic den Zustandswert approximiert und damit das Policy-Update stabilisiert. Zusätzlich wird ein antagonistischer Störterm eingeführt, der zufällige Aktionsstörungen verursacht und damit die Robustheit der gelernten Strategie gegenüber Unsicherheiten verbessert.

Das Reward-Design kombiniert mehrere Komponenten: ein phasenabhängiges Flächenwachstum der geschätzten Ellipse als Hauptziel, eine Strafe für degenerierte Ellipsenformen, eine Energie-Strafe für übermäßige Gelenkbewegungen sowie eine Terminalbewertung zur Förderung einer geschlossenen Trajektorie. Durch dieses Design wird der Agent dazu angeleitet, stabile, große und geometrisch gültige Ellipsenbewegungen zu erzeugen.

## 1) Ziel des Projektes

- hohe finale Ellipsenfläche,
- gute Schließung der Trajektorie,
- nicht-degenerierte Ellipse (Achsenverhältnis),
- robust gegenüber antagonistischer Störung (`adv_noise_scale`).

---

## 2) Verzeichnisüberblick

```text
scripts/
├── __init__.py
├── finger_env.py                 # Gymnasium-Environment + Reward + Render
├── actor_critic.py               # Actor/Critic Modelle
├── train.py                      # Single-Env Training
├── rollout.py                    # Single-Env Rollout
├── evaluate.py                   # Modell laden, evaluieren, rendern
├── visualize_results.py          # Lernkurven/Robustheit/Trajektorie/ plotten
├── requirements.txt

```

---

## 3) Usage
### Schritt 0 - Voraussetzungen

- Python 3.10+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
```bash
pip install -r scripts/requirements.txt
```

### Schritt 2 – Environment spielen

Das Environment enthält mehrere Hyperparameter, die in Punkt 4.1 detailliert erklärt sind.
Die Anzahl der Environment-Schritte pro Episode wird über den Hyperparameter `horizon` in der Klasse `EnvConfig` gesteuert.


### Schritt 3 – Ergebnisse visualisieren

Datei: `scripts/visualize_results.py`

Start:

```bash
python scripts/visualize_results.py
```

Erzeugt folgende Plots:
- finale Trajektorie + PCA-Ellipse,
- Lernkurven aus `training_log.pt` (oder aus gespeicherten PNGs),
- Robustheitsplots gegen `adv_noise_scale`.

Parameter:

| Name | Wert | Bedeutung |
|---|---|---|
| `SEED` | `0` | Zufallsseed für reproduzierbare Auswertung/Visualisierung. |
| `EVALUATE_EPISODES_NUM` | `50` | Anzahl der Episoden, über die die Metriken für die Visualisierung gemittelt werden. |
| `CHECKPOINT` | `checkpoints/20446-episoden/best_by_eval_return.pt` | Pfad zum zu ladenden Modell-Checkpoint für Trajektorie, Lernkurven und Robustheitsplots. |

### Schritt 4 – Modell evaluieren oder rendern

Datei: `scripts/evaluate.py`

Oben im Skript einstellen:
- `MODE = "render"` oder `"eval"`
- Checkpoint-Pfade
- Noise-Level-Liste (`adv_noise_scale`)

Start:

```bash
python scripts/evaluate.py
```

- `render`: spielt 1 Episode im Environment sichtbar ab (`render_mode="human"`)
- `eval`: mehrere Noise-Stufen und Ausgabe von Return/Area/Closure/Axis Ratio


### Schritt 5 – Training starten

Datei: `scripts/train.py`

Dort werden oben im Skript die wichtigsten Laufparameter gesetzt:
- `EPISODEN`
- `MODE` (`"NEU"` oder `"RESUME"`)
- ggf. `RESUME_PATH`
- Lernraten / Entropy / Value-Koeffizient

Start:

```bash
python scripts/train.py
```

Output (in `checkpoints/...`):
- `best_by_eval_return.pt`
- `best_by_eval_area.pt`
- `last_model.pt`
- `training_log.pt`
- `training_curve_return.png`
- `training_curve_area.png`
- `training_curve_losses.png`
- `training_curve_entropy.png`





### Häufige Fehler

1. **Falscher `MODE` im Training**
   - In `train.py` muss `MODE` exakt `"NEU"` oder `"RESUME"` sein.

2. **Ungültiger Checkpoint-Pfad bei Resume/Eval**
   - Prüfe `RESUME_PATH` bzw. Pfade in `evaluate.py`.

3. **Render funktioniert nicht auf Headless-System**
   - `render_mode="human"` braucht grafische Oberfläche.

4. **Importprobleme beim direkten Ausführen aus Unterordnern**
   - Skripte vom Projekt-Root aus starten (`python scripts/...`).

5. **Unklare Vergleichbarkeit von Runs**
   - Konfigurationen, Seeds und Noise-Level pro Run protokollieren.

---

## 4) Projektstruktur und Komponenten

### 1  `finger_env.py`

`FingerEllipseEnv` ist ein kontinuierliches Gymnasium-Environment für einen planaren 3R-Finger.
Ziel ist eine große, geschlossene und nicht-degenerierte Endeffektor-Trajektorie (Ellipse-ähnlich).

- **Fläche maximieren:** Die Trajektorie soll eine große Ellipsenfläche aufspannen.
- **Schließung erzwingen:** Start- und Endpunkt sollen nahe beieinander liegen.
- **Degeneration vermeiden:** Die Ellipse soll nicht zu „flach“ werden (`b/a` nicht zu klein).
- **Störrobustheit:**  antagonistisches Rauschen auf Aktionen.

#### EnvConfig

Die zentralen Environment-Parameter stehen in `scripts/finger_env.py` in der Klasse `EnvConfig`:

| Parameter | Wert | Bedeutung |
|---|---:|---|
| `device` | `cuda:0` falls verfügbar, sonst `cpu` | Rechen-Device für Tensoren/Modelle. |
| `l1` | `5.0` | Länge Link 1 in cm. |
| `l2` | `2.5` | Länge Link 2 in cm. |
| `l3` | `2.5` | Länge Link 3 in cm. |
| `theta_min` | `-π/2` | Untere Gelenkgrenze in rad. |
| `theta_max` | `+π/2` | Obere Gelenkgrenze in rad. |
| `horizon` | `256` | Episodenlänge (maximale Schritte pro Episode). |
| `max_delta` | `0.05` | Max. Gelenkwinkel-Änderung pro Step in rad (≈ 2.8°). |
| `k_axis` | `1.0` | Skalenfaktor für PCA-Ellipsenachsen. |
| `w_area` | `1.0` | Gewicht für Flächenanteil im dense Reward (im aktuellen Reward-Code nicht explizit multipliziert). |
| `w_close` | `0.05` | Gewicht der terminalen Closure-Strafe. |
| `w_close_dense` | `0.05` | Gewicht des dense Closure-Terms während der Episode. |
| `w_degen` | `0.1` | Gewicht der terminalen Degenerationsstrafe (kleines Achsenverhältnis). |
| `w_degen_dense` | `0.01` | Gewicht der dense Degenerationsstrafe während der Episode. |
| `w_action` | `0.02` | Gewicht der Aktionsenergie-Strafe (`||a||²`) pro Step. |
| `min_axis_ratio` | `0.35` | Mindestwert für `b/a` (Ellipse), darunter greift Degenerations-Hinge. |
| `adv_noise_scale` | `0.25` | Stärke der antagonist. Störung relativ zu `max_delta`; effektive Noise-Amplitude = `adv_noise_scale * max_delta` (`0.0125` rad). |

#### Action Space

- Typ: `Box(shape=(3,), dtype=float32)`
- Bereich je Gelenk: `[-max_delta, +max_delta]`
- Bedeutung: Aktion ist **Gelenkwinkel-Delta pro Step** in rad.

![Action Space](figures/action_space.png)

Im `step()`:
1. Aktion wird auf `[-max_delta, +max_delta]` geclippt.
2. Optionales Noise wird addiert (`adv_noise_scale * max_delta * U[-1,1]`) und erneut geclippt.
3. Gelenkwinkel werden aktualisiert und auf `[theta_min, theta_max]` geclippt.

![Action Clipping](figures/Action_clippen.png)

#### Observation Space

- Typ: `Box(shape=(9,), low=-1, high=1, dtype=float32)`
- Struktur (Reihenfolge):

| Index | Feature | Beschreibung |
|---:|---|---|
| 0 | `sin(theta1)` | Sinus von Gelenk 1 |
| 1 | `cos(theta1)` | Cosinus von Gelenk 1 |
| 2 | `sin(theta2)` | Sinus von Gelenk 2 |
| 3 | `cos(theta2)` | Cosinus von Gelenk 2 |
| 4 | `sin(theta3)` | Sinus von Gelenk 3 |
| 5 | `cos(theta3)` | Cosinus von Gelenk 3 |
| 6 | `x_norm` | Endeffektor-x, normiert mit `l1+l2+l3` |
| 7 | `y_norm` | Endeffektor-y, normiert mit `l1+l2+l3` |
| 8 | `phase` | normierte Zeit `t/horizon` |

![Observation Space](figures/observation.png)


#### Geometriebasis (Ellipse aus Punktwolke)

Aus der bisherigen Trajektorie wird über die Kovarianzmatrix $\Sigma$ eine PCA-Ellipse geschätzt:

$$
A = \pi \cdot k_{axis}^2 \cdot \sqrt{\det(\Sigma)}
$$

Dabei sind $a, b$ die Halbachsen und `axis_ratio = b/a`.
`k_axis` aus `EnvConfig` hat standardmäßig den Wert **1.0**.

#### Reward-Funktion (Dense + Terminal)

Die Reward-Berechnung in `reward_function()` ist **phasenabhängig** und besteht aus Dense Terms pro Step plus terminaler Penalty am Episodenende.

##### 1) Dense Reward-Terme

**(a) Flächenterm (Area)**

$$
r_{area}(t)=\alpha_{area}(t)\cdot(A_t-A_{t-1})
$$

mit

$$
\alpha_{area}(t)=\max\left(0.2,\;1-\max\left(0,\frac{\text{phase}-0.8}{0.2}\right)\right)
$$



- Bis 80% der Episode: Gewicht gleich 1.0
- Danach linearer Abfall, aber nur bis 0.2

![Area Weighting](reward_plotten/flächen_wachstum.png)

**(b) Aktionskosten (Action)**

$$
r_{action}(t)=w_{action}\cdot\|a_t\|^2
$$

Mit `w_action = 0.02` . Dieser Term wird vom Reward abgezogen.

![Action Penalty](reward_plotten/dense_action_penalty.png)

**(c) Degenerationsstrafe (Degeneration)**

$$
   \mathrm{hinge}=\max(0,\;\mathrm{minAxisRatio}-b/a)
$$

$$
r_{degen}(t)=w_{\mathrm{degenDense}}\cdot\alpha_{degen}(t)\cdot\mathrm{hinge}^2
$$

mit

$$
\alpha_{degen}(t)=\max\left(0,\frac{\text{phase}-0.25}{0.75}\right)
$$

Mit `min_axis_ratio = 0.35` und `w_degen_dense = 0.01`.


![Degeneration Weighting](reward_plotten/dense_degenration_penalty.png)

**(d) Closure-Term (Closure, späte Phase)**

$$
r_{close}(t)=w_{close\_dense}\cdot\alpha_{close}(t)\cdot(d_{t-1}-d_t)
$$

mit $d_t=\|p_t-p_0\|^2$ und

$$
\alpha_{close}(t)=\max\left(0,\frac{\text{phase}-0.7}{0.3}\right)
$$

Mit `w_close_dense = 0.05` .


![Closure Weighting](reward_plotten/dense_closure_reward.png)

**Gesamter Dense Reward:**

`reward_dense = r_area_dense - r_action_dense + r_close_dense - r_degen_dense`


**Interpretation (Phasenverlauf):**

Diese Grafik zeigt die zeitliche Struktur der Reward-Funktion während einer Episode.

- **Phase 0–25 %**
   - Der Agent konzentriert sich vollständig auf Flächenwachstum.
   - Area-Reward ist maximal.
   - Keine Degeneration-Strafe.
   - Keine Closure-Belohnung.
   - **Ziel:** Exploration großer Trajektorien.

- **Phase 25–70 %**
   - Die Degeneration-Strafe wird schrittweise aktiviert.
   - Das verhindert, dass die Trajektorie zu einer Linie degeneriert oder extrem gestauchte Ellipsen erzeugt.
   - Der Agent maximiert weiter Fläche, muss aber gleichzeitig eine gültige Ellipsenform erhalten.

- **Phase 70–80 %**
   - Die Closure-Belohnung beginnt.
   - Der Agent wird gefördert, zum Startpunkt zurückzukehren und die Trajektorie zu schließen.
   - Die Flächengewichtung ist weiterhin maximal.

- **Phase 80–100 %**
   - Die Flächengewichtung wird reduziert.
   - Dadurch verschiebt sich die Priorität: weniger Fokus auf zusätzliche Fläche, stärkerer Fokus auf das saubere Schließen der Ellipse.


Für die Gesamtübersicht der Phasen-Gewichte (`alpha_area`, `alpha_degen`, `alpha_close` 

![Reward Phase Weights](reward_plotten/reward_design_phase_weights.png)

##### 2) Terminale Penalty

Am Episodenende (`truncated` oder `terminated`) wird zusätzlich abgezogen:

$$
p_{close}=w_{\mathrm{close}}\cdot\mathrm{closureDist2}
$$

Mit `w_close = 0.05` .

$$
p_{degen}=w_{\mathrm{degen}}\cdot\mathrm{hinge}^2
$$

Mit `w_degen = 0.1` .

$$
   \mathrm{terminalPenalty}=p_{close}+p_{degen}
$$

Final im letzten Schritt:

`reward = reward_dense - terminal_penalty`



#### API-Verhalten (`reset` / `step`)

- `reset(seed=...)`
  - initialisiert zufällige Gelenkwinkel innerhalb Limits,
  - setzt interne Trajektorie auf Startpunkt,
  - gibt `(obs, info)` zurück.

- `step(action)`
  - führt Dynamik + Reward aus,
  - aktualisiert Trajektorie und Zeit,
  - gibt `(obs, reward, terminated, truncated, info)` zurück.

#### Rendering

- Bei `render_mode="human"` wird pro Step ein Matplotlib-Fenster aktualisiert.
- Gezeigt werden:
  - Finger-Glieder als Polyline,
  - Endeffektor-Trajektorie,
  - feste Achsenskalierung auf den Arbeitsraum `l1+l2+l3`.

### 2 `actor_critic.py`

- Implementiert die Policy (Actor) und die Zustandswertfunktion (Critic).

#### Actor

- **Eingabe:** Observation mit `obs_dim = 9`.
- **Netzwerk:** `Linear(9, 128) -> Tanh -> Linear(128, 128) -> Tanh`.
- **Ausgabe 1 (Mittelwert):** `mu_head: Linear(128, act_dim)` mit `act_dim = 3` (für `Δθ1, Δθ2, Δθ3`).
- **Ausgabe 2 (Std-Abw.):** globaler, lernbarer Parameter `log_std` mit Form `(3,)`; daraus `std = exp(log_std)`.
- **Verteilung:** Gauß-Verteilung `Normal(mu, std)` pro Aktionsdimension.
- **Sampling:** Reparameterisierung via `rsample()` und danach `tanh`-Squashing auf `[-1, 1]`.
- **choose_action(...):** gibt `(action, log_prob, entropy)` zurück.

| Größe | Form (ein Zustand)
|---|---|
| `obs` | `(9,)` | 
| `mu` | `(3,)` |
| `std` | `(3,)` |
| `action = tanh(raw_action)` | `(3,)` |

#### Critic

- **Eingabe:** Observation mit `obs_dim = 9`.
- **Netzwerk:** `Linear(9, 128) -> Tanh -> Linear(128, 128) -> Tanh -> Linear(128, 1)`.
- **Ausgabe:** Zustandswert `V(s)` als Skalar (durch `squeeze(-1)`).

| Größe | Form (ein Zustand)
|---|---|
| `obs` | `(9,)` |
| `V(s)` | `()` |

### 3 `rollout.py`

- Sammelt Trainingsdaten für ein Actor-Critic-Update über maximal `horizon = 256` Schritte.

#### Ablauf in `collect_rollout(...)`

1. `env.reset()` liefert Startzustand `obs`.
2. Pro Schritt:
   - Actor erzeugt Aktion, Log-Prob und Entropie.
   - Critic schätzt `V(s)`.
   - Aktion wird von `[-1,1]` auf `[-max_delta, +max_delta]` skaliert.
   - `env.step(action)` liefert `next_obs, reward, terminated, truncated, info`.
   - Alle Größen werden in Listen gespeichert.
3. Bei `terminated` oder `truncated` wird der Rollout beendet.
4. Listen werden in Tensoren umgewandelt und als `rollout`-Dictionary zurückgegeben.

#### Rückgabe-Tensoren

| Key | Bedeutung | Form |
|---|---|---|
| `obs` | Beobachtungen | `(T, 9)` |
| `actions` | ausgeführte Aktionen (skaliert) | `(T, 3)` |
| `rewards` | Rewards pro Schritt | `(T,)` |
| `log_probs` | Log-Wahrscheinlichkeiten der Aktionen | `(T,)` |
| `values` | Critic-Schätzung `V(s_t)` | `(T,)` |
| `entropies` | Policy-Entropie pro Schritt | `(T,)` |
| `dones` | Episodenende-Flags (`0/1`) | `(T,)` |
| `last_info` | letztes `info` aus dem Env | `dict` |

`T` ist die tatsächlich gelaufene Schrittlänge und erfüllt `1 <= T <= horizon`.

### 4 `train.py`

- Monte-Carlo Returns
- Advantage-Normalisierung
- getrennte Optimierer für Actor/Critic
- regelmäßige Evaluation während des Trainings über eine bestimmte Anzahl von Episoden mithilfe von `evaluate(...)`
- Speichern von Best-/Last-Checkpoints + Trainingslog + Kurvenplots

### 5 `evaluate.py`
- Zwei Modi über den Parameter `MODE`: `"render"` oder `"eval"`.
- `render`-Mode: spielt genau **eine** Episode mit den aktuellen Hyperparametern sichtbar ab.
- `eval`-Mode: evaluiert die Policy über mehrere Episoden (z. B. `EVALUATE_EPISODES_NUM = 50`) und über mehrere Noise-Level.
- Im `eval`-Mode können beliebig viele Policies/Checkpoints gemeinsam evaluiert werden (Liste `checkpoints = [...]`).
- Lädt den Checkpoint (`actor_state_dict`, `critic_state_dict`, `config`) und nutzt deterministische Aktionen über `tanh(mu)`.
- Im `eval`-Mode werden Metriken (Return, Area, Closure, Axis Ratio) aggregiert und über `visualize_results.py` mit `plot_eval_records(...)` gegeneinander geplottet.

### 6 `visualize_results.py`

- Lernkurven aus Logs
- Trajektorie inkl. PCA-Ellipse
- Robustheit über Noise-Level

---



## 5) Definition von Testszenarien vor dem Training

Für die Bewertung der Policy werden vor dem Training folgende Testszenarien definiert:

1. **Robustheit-Szenario (Störrauschen)**
   - Parameter: `adv_noise_scale = [0.0, 0.1, 0.25, 0.5, 0.75]`
   - Fragestellung: Wie robust ist die gelernte Policy gegenüber antagonistischer Störung auf den Aktionen?

2. **Evaluierungs-Episoden**
   - Parameter: `EVALUATE_EPISODES_NUM = [20, 50, 100]`
   - Fragestellung: Wie stabil sind die gemittelten Metriken (Return, Fläche, Closure, Axis Ratio) in Abhängigkeit von der Anzahl der Evaluierungs-Episoden?

3. **Startzustand über Seeds**
   - Parameter: `SEED = [0, 2, 5]`
   - Fragestellung: Wie stark hängt die Performance vom initialen Startzustand bzw. der Zufallsinitialisierung ab?

4. **Änderung der Episodenlänge**
   - Parameter: `T = [T, T/2, 2T]` mit `T = horizon` aus `EnvConfig`
   - Fragestellung: Wie verändert sich das Verhalten bei kürzeren/längeren Episoden?
   - Wichtig: Die Phase `t/T` ist Teil des Observation Space (`phase`) und beeinflusst damit direkt die Policy-Eingabe.


## 6) Hyperparameter-Studie

Die Hyperparameter-Studie besteht aus zwei zentralen Teilen: Trainingsparameter und Environment-Parameter.

### Training-Parameter

Die final verwendeten Training-Hyperparameter (aus `scripts/train.py`) sind:

| Name | Wert | Bedeutung |
|---|---:|---|
| `EPISODEN` | `20500` | Anzahl der Trainings-Episoden. |
| `GAMMA` | `0.99` | Diskontfaktor für Monte-Carlo-Returns. |
| `ACTOR_LR` | `0.0003` | Lernrate des Actor-Optimierers. |
| `CRITIC_LR` | `0.0003` | Lernrate des Critic-Optimierers. |
| `hidden_dim` | `128` | Breite der Hidden Layer in Actor und Critic. |
| `ENTROPY_COEF` | `0.0025` | Gewicht des Entropie-Bonus zur Förderung von Exploration. |
| `VALUE_COEF` | `0.5` | Gewichtung des Critic-Loss im Value-Update. |

**`CRITIC_LR`**

Zur Stabilisierung bzw. für eine schnellere Reaktion der Value-Schätzung wurde testweise `CRITIC_LR = 0.0001` verwendet.
Dabei zeigte sich jedoch, dass die Entropie im Training zu schnell in den negativen Bereich fällt, wodurch die Policy zu früh deterministisch wird (zu wenig Exploration). Daher wurde `CRITIC_LR` auf `0.0003` gesetzt.

**`ENTROPY_COEF`**

Zu Beginn wurde `ENTROPY_COEF = 0.001` verwendet, was zu einem Entropie-Kollaps führte.
Die Entropie fiel sehr schnell in den negativen Bereich, wodurch keine ausreichende Exploration der Policy mehr stattfand. Mit `ENTROPY_COEF = 0.0025` nähert sich die Entropie dem Wert 0 erst nach ungefähr 19.000 Episoden.

**`hidden_dim`**

Die Größe der neuronalen Netze für Actor und Critic orientiert sich an der bekannten Referenzumgebung CartPole. Diese besitzt eine Zustandsdimension von vier und einen diskreten Aktionsraum mit zwei möglichen Aktionen. Da die vorliegende Umgebung des anthropomorphen Fingers eine deutlich höhere Zustandsdimension (9 Beobachtungsvariablen) sowie einen kontinuierlichen Aktionsraum mit drei Dimensionen besitzt, wurde die Netzarchitektur entsprechend erweitert. Daher wurden für Actor und Critic jeweils zwei Hidden Layers mit einer Breite von 128 Neuronen gewählt, um ausreichend Modellkapazität bei gleichzeitig stabilem Training zu gewährleisten.

### Environment-Parameter
**`w_area`**
Da das Hauptziel des Projekts darin besteht, eine möglichst große Ellipse zu erzeugen, wird dieser Parameter auf `1.0` gesetzt. Dadurch erhält der Flächenanteil den größten Einfluss innerhalb der Reward-Funktion.


**`w_close_dense`**
w_close_dense gewichtet die dichte Strafe für eine nicht geschlossene Trajektorie. Dadurch wird der Agent dazu angehalten, eine periodische Bewegung zu erzeugen, die zum Startpunkt zurückführt, sodass eine konsistente Ellipse besser eingeschlossen werden kann.

Zu Beginn wurde der Parameter auf 0.01 gesetzt, was sich als zu klein erwies, da die Strafe für eine offene Trajektorie kaum Einfluss auf das Lernverhalten hatte. Anschließend wurde der Wert auf 0.1 erhöht, was sich als zu aggressiv herausstellte: Der Agent reduzierte seine Bewegung stark und tendierte dazu, nahezu still zu bleiben, um Strafen zu vermeiden.

Daher wurde schließlich ein Mittelwert von 0.05 gewählt. Dieser Wert stellt einen sinnvollen Kompromiss dar, da er den Agenten ausreichend dazu motiviert, die Trajektorie zu schließen, ohne dabei die Bewegung zu stark zu unterdrücken.

**`w_action`**
Der Parameter `w_action` gewichtet die Strafe für große Aktionsänderungen und dient als Regularisierung der Bewegung. Dadurch wird der Agent dazu angehalten, gleichmäßigere und energieeffizientere Bewegungen auszuführen, anstatt sehr große oder abrupte Gelenkänderungen zu erzeugen.
Bei einem Wert von 0.05 tendierte der Agent hier auch dazu, still zu bleiben.

**`min_axis_ratio`**
Der Parameter `min_axis_ratio` definiert das minimale zulässige Verhältnis der beiden Ellipsenhalbachsen `b/a` und dient dazu, stark degenerierte Ellipsen zu vermeiden. Eine ideale Ellipsenform in dieser Aufgabe weist typischerweise ein Achsenverhältnis von etwa `0.5` auf, während sehr kleine Werte darauf hinweisen, dass die Trajektorie eher linienförmig als elliptisch ist.

Zu Beginn wurde der Parameter auf `0.25` gesetzt. Dieser Wert erwies sich jedoch als zu niedrig, da der Agent Ellipsen mit einem Achsenverhältnis von etwa `0.35` erzeugte, die zwar gültig waren, jedoch teilweise noch zu stark verzerrt waren.