import os
import re
import random
import csv
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# -------------------------
# SETTINGS
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "safe_syntetic_faces")

N_REPEATS_PER_CONDITION = 5   # 2 endpoint x 3 test stimuli x N_REPEATS = trial totali
ADAPT_DURATION_RANGE = (20, 30)   # secondi
TEST_DURATION_RANGE = (0.5, 1.0)  # secondi
REST_DURATION = 5.0               # secondi tra un trial e l'altro
NEUTRAL_OFFSET_FRACTION = 0.15    # offset dei 3 test stimuli rispetto al neutro

DISPLAY_SIZE = (500, 500)

student_id = input("Enter student ID: ")
OUTPUT_FILE = f"{student_id}_exp3.csv"

# -------------------------
# LOAD IMAGES + PREDICTED RATING
# -------------------------

def extract_predicted_rating(filename):
    match = re.search(r"rating_([0-9]*\.?[0-9]+)", filename)
    if match is None:
        raise ValueError(f"Impossibile estrarre il predicted rating da: {filename}")
    return float(match.group(1))

all_images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_ratings = {f: extract_predicted_rating(f) for f in all_images}

print("Numero di immagini disponibili:", len(all_images))

# -------------------------
# SELEZIONE STIMOLI: 2 ENDPOINT + 3 TEST VICINO AL NEUTRO
# -------------------------

sorted_images = sorted(image_ratings.items(), key=lambda x: x[1])

unhappy_endpoint_file, unhappy_endpoint_val = sorted_images[0]
happy_endpoint_file, happy_endpoint_val = sorted_images[-1]

neutral_val = (unhappy_endpoint_val + happy_endpoint_val) / 2
offset = NEUTRAL_OFFSET_FRACTION * (happy_endpoint_val - unhappy_endpoint_val)

target_values = [neutral_val - offset, neutral_val, neutral_val + offset]

excluded = {unhappy_endpoint_file, happy_endpoint_file}
test_files = []

for target in target_values:
    candidates = [
        (f, v) for f, v in image_ratings.items()
        if f not in excluded and f not in test_files
    ]
    closest_file = min(candidates, key=lambda x: abs(x[1] - target))[0]
    test_files.append(closest_file)

print("Endpoint unhappy:", unhappy_endpoint_file, unhappy_endpoint_val)
print("Endpoint happy:", happy_endpoint_file, happy_endpoint_val)
print("Test stimuli:", [(f, image_ratings[f]) for f in test_files])

adaptors = [
    ("unhappy", unhappy_endpoint_file, unhappy_endpoint_val),
    ("happy", happy_endpoint_file, happy_endpoint_val),
]

# -------------------------
# BUILD TRIAL LIST (6 condizioni x N_REPEATS)
# -------------------------

trials = []

for adaptor_condition, adaptor_file, adaptor_val in adaptors:
    for test_file in test_files:
        for _ in range(N_REPEATS_PER_CONDITION):
            trials.append({
                "adaptor_condition": adaptor_condition,
                "adaptor_file": adaptor_file,
                "adaptor_rating": adaptor_val,
                "test_file": test_file,
                "test_rating": image_ratings[test_file],
                "adapt_duration": random.uniform(*ADAPT_DURATION_RANGE),
                "test_duration": random.uniform(*TEST_DURATION_RANGE),
                "rating": None,
            })

random.shuffle(trials)

print("Numero totale di trial:", len(trials))

# -------------------------
# GUI ROOT (deve esistere PRIMA di creare qualsiasi PhotoImage)
# -------------------------

root = tk.Tk()
root.title("Face Rating Experiment - Part 3 (Adaptation)")

# -------------------------
# PRE-CARICAMENTO IMMAGINI CON FISSAZIONE
# -------------------------

def add_fixation(img):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    size = 12
    draw.line((cx - size, cy, cx + size, cy), fill="red", width=3)
    draw.line((cx, cy - size, cx, cy + size), fill="red", width=3)
    return img

photo_cache = {}

unique_stimulus_files = {unhappy_endpoint_file, happy_endpoint_file, *test_files}

for filename in unique_stimulus_files:
    path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(path)
    img.thumbnail(DISPLAY_SIZE)
    img = add_fixation(img)
    photo_cache[filename] = ImageTk.PhotoImage(img)

# schermata grigia neutra con sola fissazione (usata tra la fine del test e la risposta, e nel riposo)
blank_img = Image.new("RGB", DISPLAY_SIZE, (128, 128, 128))
blank_img = add_fixation(blank_img)
photo_cache["__BLANK__"] = ImageTk.PhotoImage(blank_img)

# -------------------------
# EXPERIMENT STATE MACHINE
# -------------------------

current_trial_index = 0


def show_image(key):
    image_label.config(image=photo_cache[key])
    image_label.image = photo_cache[key]


def hide_buttons():
    button_frame.pack_forget()


def show_buttons():
    button_frame.pack(pady=10)


def run_trial(index):
    global current_trial_index
    current_trial_index = index

    trial = trials[index]

    hide_buttons()
    show_image(trial["adaptor_file"])
    progress_label.config(
        text=f"Trial {index + 1}/{len(trials)} - Adaptation ({trial['adaptor_condition']})"
    )

    adapt_ms = int(trial["adapt_duration"] * 1000)
    root.after(adapt_ms, lambda: show_test(index))


def show_test(index):
    trial = trials[index]

    show_image(trial["test_file"])
    progress_label.config(text=f"Trial {index + 1}/{len(trials)} - Test")

    test_ms = int(trial["test_duration"] * 1000)
    root.after(test_ms, lambda: prompt_response(index))


def prompt_response(index):
    show_image("__BLANK__")
    progress_label.config(text=f"Trial {index + 1}/{len(trials)} - Rate the face")
    show_buttons()


def save_rating(rating):
    trial = trials[current_trial_index]
    trial["rating"] = rating

    hide_buttons()

    next_index = current_trial_index + 1

    if next_index < len(trials):
        show_image("__BLANK__")
        progress_label.config(text="Rest...")
        root.after(int(REST_DURATION * 1000), lambda: run_trial(next_index))
    else:
        finish_experiment()


def finish_experiment():

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "trial_number",
            "adaptor_condition",
            "adaptor_file",
            "adaptor_predicted_rating",
            "test_file",
            "test_predicted_rating",
            "adapt_duration_s",
            "test_duration_s",
            "rating",
        ])

        for i, trial in enumerate(trials):
            writer.writerow([
                i + 1,
                trial["adaptor_condition"],
                trial["adaptor_file"],
                trial["adaptor_rating"],
                trial["test_file"],
                trial["test_rating"],
                round(trial["adapt_duration"], 2),
                round(trial["test_duration"], 2),
                trial["rating"],
            ])

    print("Experiment finished!")
    print("Data saved to:", OUTPUT_FILE)

    root.destroy()


# -------------------------
# RESTO DELLA GUI
# -------------------------

instruction_label = tk.Label(
    root,
    text="How happy does this person look?\n1 = Very unhappy    5 = Very happy",
    font=("Arial", 18)
)
instruction_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

button_frame = tk.Frame(root)

for rating in range(1, 6):
    button = tk.Button(
        button_frame,
        text=str(rating),
        font=("Arial", 18),
        width=4,
        command=lambda r=rating: save_rating(r)
    )
    button.pack(side=tk.LEFT, padx=5)

progress_label = tk.Label(root, text="", font=("Arial", 12))
progress_label.pack(pady=10)

run_trial(0)

root.mainloop()
