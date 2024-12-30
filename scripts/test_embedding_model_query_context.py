import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from settings import EMBEDDER_MODEL, NORMALIZE_EMBEDDINGS
from src.embedder import Embedder
import numpy as np

# Prepare test data
questions = [
    "Was ist ein Dezibel?",
    "Wie errechnet sich die Höhe des Solidaritätszuschlags?",
    "Was bildet die Rechtsgrundlage für die Erhebung der Gemeindesteuern?",
    "Wie ist die politische Ausstattung des Bundespräsidenten im Grundgesetz?",
    "Wozu dient die Gewerbesteuer?",
]

contexts = [
    """Das Bel (Einheitenzeichen B) ist eine Hilfsmaßeinheit zur Kennzeichnung des dekadischen Logarithmus des Verhältnisses zweier Größen der gleichen Art bei Pegeln und Maßen.[1] Diese werden in der Elektrotechnik und der Akustik angewendet, beispielsweise bei der Angabe eines Dämpfungsmaßes oder Leistungspegels.""",
    """Der Solidaritätszuschlag beträgt 5,5 % der Einkommen- bzw. Körperschaftsteuer (§ 4 SolzG). Es gilt für die Erhebung auf die Lohn- und Einkommensteuer eine Freigrenze mit Gleitzone. Der Grenzsteuersatz (bezogen auf den Steuerbetrag) innerhalb dieser Gleitzone liegt durch die gesetzliche Berechnungsvorschrift bei 11,9 %. Danach sinkt er auf den Durchschnittssatz von 5,5 %.[2][3] Der Grenzsteuersatz für die Summe aus Einkommensteuer plus Solidaritätszuschlag (bezogen auf das zu versteuernde Einkommen, abgekürzt zvE) liegt innerhalb der Gleitzone bei 47 %, sinkt danach auf 44,31 % und steigt wieder auf 47,475 % ab dem Beginn der Tarifzone des Höchststeuersatzes.""",
    """In Art. 106 Abs. 6 GG ist vorgeschrieben, dass das Aufkommen der Gemeindesteuern den Gemeinden zusteht. Im weiteren Sinne gehören neben diesen Gemeindesteuern auch die Anteile der Kommunen an Lohnsteuer und an veranlagter Einkommensteuer sowie Anteile an Kapitalertragsteuer und Umsatzsteuer zur Gemeindesteuer. \n\nDie Gemeindesteuern wurden erst im Zuge der Finanzreform 1956 in das Grundgesetz aufgenommen. Der Bund darf diese Steuern nicht grundsätzlich entziehen, er kann aber das Aufkommen durch Gesetz einschränken. Als Beispiel seien das Steueränderungsgesetz 1979, welches das Aufkommen der Gewerbesteuer begrenzte, und die Streichung der Gewerbekapitalsteuer oder die Erhöhung der Gewerbesteuerumlage genannt""",
    """Die geringe machtpolitische Ausstattung des Amtes des Bundespräsidenten im Grundgesetz für die Bundesrepublik Deutschland gilt allgemein als eine Reaktion auf die Erfahrungen mit dem Amt des Reichspräsidenten in der Weimarer Republik.[38] Während der Beratungen des Parlamentarischen Rates herrschte weitgehender Konsens aller Beteiligten, dass dem Präsidenten nicht wieder eine solch überragende Stellung im politischen System zukommen sollte wie seinerzeit dem Reichspräsidenten (insbesondere Paul von Hindenburg).[39]\n\nParallel zu dieser Schmälerung seiner Befugnisse wurde auch der Wahlmodus für den Präsidenten verändert: Wurde der Reichspräsident noch vom Volk direkt gewählt (1925 und 1932), so wird der Bundespräsident von der nur für diesen Zweck zusammentretenden Bundesversammlung gewählt. Hierdurch wurde die demokratische Legitimation des Bundespräsidenten indirekter: Er ist nicht mehr unmittelbar vom Souverän gewähltes Organ der politischen Staatsführung. Die Ablehnung einer Direktwahl des Bundespräsidenten wird auch damit begründet, dass sonst ein Missverhältnis zwischen starker demokratischer Legitimation (er wäre dann neben dem Bundestag das einzige direkt gewählte Verfassungsorgan des Bundes,[40] zudem das einzige, das aus einer Person besteht) und geringer politischer Macht einträte.""",
    """Die Gewerbesteuer (Abkürzung: GewSt) wird als Gewerbeertragsteuer auf die objektive Ertragskraft eines Gewerbebetriebes erhoben. Hierzu wird für gewerbesteuerliche Zwecke ein Gewerbeertrag ermittelt, welcher regelmäßig in einem Gewerbesteuermessbetrag in Höhe von 3,5 % des Gewerbeertrags mündet. Die hebeberechtigte Gemeinde muss die Gewerbesteuer mindestens in Höhe des doppelten Messbetrages erheben (Hebesatzminimum: 200 %).\n\nEine ertragsunabhängige Besteuerung der Substanz des Gewerbebetriebs erfolgte bis 1997 mit der Gewerbekapitalsteuer, seitdem nur noch in den Gewinnhinzurechnungen, die bestimmte Finanzierungskosten in die gewerbesteuerliche Bemessungsgrundlage einbeziehen. Mit der Unternehmensteuerreform 2008 wurde diese Komponente ausgeweitet, um das Gewerbesteueraufkommen zu verstetigen."""
]

# Load the embedding model
embedder = Embedder(EMBEDDER_MODEL, normalize=NORMALIZE_EMBEDDINGS)

def cosine_similarity(a, b):
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_matrix = np.zeros((len(questions), len(contexts)))
for i, question in enumerate(questions):
    question_embedding = embedder.embed([question])[0]
    for j, context in enumerate(contexts):
        context_embedding = embedder.embed([context])[0]
        similarity_matrix[i, j] = cosine_similarity(question_embedding, context_embedding)

match_similarity = np.trace(similarity_matrix) / len(questions)
print(f"Match similarity: {match_similarity:.2f}")

# Get off-diagonal elements
no_match_similarity = (np.sum(similarity_matrix) - np.trace(similarity_matrix)) / (len(questions) * len(contexts) - len(questions))
print(f"No match similarity: {no_match_similarity:.2f}")

# Ratio of match similarity to no match similarity
# This metric measures the quality of the embeddings
ratio = match_similarity / no_match_similarity
print(f"Ratio of match similarity to no match similarity: {ratio:.2f}")