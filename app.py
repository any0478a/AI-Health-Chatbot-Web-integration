import os, re, random, csv, warnings
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
# ✅ FIX 1: Cookie-based session only — no Flask-Session / filesystem needed
app.secret_key = os.environ.get("SECRET_KEY", "H8x#kP2@mQ9zL5nR7wT4")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False

# ✅ FIX 2: Absolute paths — relative paths fail on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ Load Data ------------------
training = pd.read_csv(os.path.join(BASE_DIR, "Data", "Training.csv"))
testing  = pd.read_csv(os.path.join(BASE_DIR, "Data", "Testing.csv"))
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns  = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing  = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x    = training[cols]
y    = training["prognosis"]
le   = preprocessing.LabelEncoder()
y    = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ------------------ Load Master Data ------------------
severityDictionary   = {}
description_list     = {}
precautionDictionary = {}
symptoms_dict        = {symptom: idx for idx, symptom in enumerate(x)}

with open(os.path.join(BASE_DIR, "MasterData", "symptom_Description.csv")) as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            description_list[row[0]] = row[1]

with open(os.path.join(BASE_DIR, "MasterData", "Symptom_severity.csv")) as f:
    for row in csv.reader(f):
        try:
            severityDictionary[row[0]] = int(row[1])
        except:
            pass

with open(os.path.join(BASE_DIR, "MasterData", "symptom_precaution.csv")) as f:
    for row in csv.reader(f):
        if len(row) >= 5:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# ✅ FIX 3: Pre-compute disease->symptoms map (avoids large data in cookie session)
ALL_SYMPTOMS = list(cols)
DISEASE_SYMPTOMS_MAP = {}
for disease in training["prognosis"].unique():
    row = training[training["prognosis"] == disease].iloc[0][:-1]
    DISEASE_SYMPTOMS_MAP[disease] = list(row.index[row == 1])

symptom_synonyms = {
    "stomach ache":        "stomach_pain",
    "belly pain":          "stomach_pain",
    "tummy pain":          "stomach_pain",
    "loose motion":        "diarrhea",
    "motions":             "diarrhea",
    "high temperature":    "fever",
    "temperature":         "fever",
    "feaver":              "fever",
    "coughing":            "cough",
    "throat pain":         "sore_throat",
    "cold":                "chills",
    "breathing issue":     "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache":           "muscle_pain",
}

def extract_symptoms(user_input):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in ALL_SYMPTOMS:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in ALL_SYMPTOMS], n=1, cutoff=0.8)
        if close:
            for sym in ALL_SYMPTOMS:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = int(np.argmax(pred_proba))
    disease    = le.inverse_transform([pred_class])[0]
    confidence = round(float(pred_proba[pred_class]) * 100, 2)
    return disease, confidence

quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish.",
]

# ------------------ Routes ------------------
@app.route("/")
def index():
    session.clear()
    session["step"] = "welcome"
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    step     = session.get("step", "welcome")

    if step == "welcome":
        session["step"] = "name"
        return jsonify(reply="🤖 Welcome to HealthCare ChatBot!\n👉 What is your name?")

    elif step == "name":
        session["name"] = user_msg
        session["step"] = "age"
        return jsonify(reply="👉 Please enter your age:")

    elif step == "age":
        session["age"]  = user_msg
        session["step"] = "gender"
        return jsonify(reply="👉 What is your gender? (M/F/Other):")

    elif step == "gender":
        session["gender"] = user_msg
        session["step"]   = "symptoms"
        return jsonify(reply="👉 Describe your symptoms in a sentence:")

    elif step == "symptoms":
        symptoms_list = extract_symptoms(user_msg)
        if not symptoms_list:
            return jsonify(reply="❌ Could not detect valid symptoms. Please describe again:")
        disease, conf = predict_disease(symptoms_list)
        session["symptoms"]     = symptoms_list
        session["pred_disease"] = disease
        session["step"]         = "days"
        session.modified        = True   # ✅ FIX 4: required for list/dict mutations
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(symptoms_list)}\n👉 For how many days have you had these symptoms?")

    elif step == "days":
        session["days"]  = user_msg
        session["step"]  = "severity"
        session.modified = True
        return jsonify(reply="👉 On a scale of 1–10, how severe is your condition?")

    elif step == "severity":
        session["severity"] = user_msg
        session["step"]     = "preexist"
        session.modified    = True
        return jsonify(reply="👉 Do you have any pre-existing conditions?")

    elif step == "preexist":
        session["preexist"] = user_msg
        session["step"]     = "lifestyle"
        session.modified    = True
        return jsonify(reply="👉 Do you smoke, drink alcohol, or have irregular sleep?")

    elif step == "lifestyle":
        session["lifestyle"] = user_msg
        session["step"]      = "family"
        session.modified     = True
        return jsonify(reply="👉 Any family history of similar illness?")

    elif step == "family":
        session["family"] = user_msg
        disease           = session.get("pred_disease", "")
        disease_syms      = DISEASE_SYMPTOMS_MAP.get(disease, [])
        session["disease_syms"] = disease_syms
        session["ask_index"]    = 0
        session["step"]         = "guided"
        session.modified        = True
        return ask_next_symptom()

    elif step == "guided":
        idx          = session.get("ask_index", 1) - 1
        disease_syms = session.get("disease_syms", [])
        if 0 <= idx < len(disease_syms) and user_msg.lower() == "yes":
            syms = session.get("symptoms", [])
            syms.append(disease_syms[idx])
            session["symptoms"] = syms
            session.modified    = True
        return ask_next_symptom()

    elif step == "final":
        return final_prediction()

    # fallback — reset
    session.clear()
    session["step"] = "welcome"
    return jsonify(reply="🔄 Session reset. What is your name?")


def ask_next_symptom():
    i  = session.get("ask_index", 0)
    ds = session.get("disease_syms", [])
    if i < min(8, len(ds)):
        sym = ds[i]
        session["ask_index"] = i + 1
        session.modified     = True
        return jsonify(reply=f"👉 Do you also have {sym.replace('_', ' ')}? (yes/no):")
    else:
        session["step"]  = "final"
        session.modified = True
        return final_prediction()


def final_prediction():
    symptoms    = session.get("symptoms", [])
    disease, conf = predict_disease(symptoms)
    about       = description_list.get(disease, "No description available.")
    precautions = precautionDictionary.get(disease, [])
    name        = session.get("name", "friend")

    text = (
        f"{'─'*44}\n"
        f"🩺 You may have: **{disease}**\n"
        f"🔎 Confidence: {conf}%\n"
        f"📖 About: {about}\n"
    )
    if precautions:
        text += "\n🛡️ Suggested precautions:\n"
        text += "\n".join(f"  {i+1}. {p}" for i, p in enumerate(precautions))
    text += f"\n\n💡 {random.choice(quotes)}"
    text += f"\n\nThank you, {name}! Wishing you good health. 🌟"
    return jsonify(reply=text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
