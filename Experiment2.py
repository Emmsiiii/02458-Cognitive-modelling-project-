import os
import random
import csv
import tkinter as tk
from PIL import Image, ImageTk

# -------------------------
# SETTINGS
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "safe_syntetic_faces")

N_REPEATS = 10  # ogni immagine presentata almeno 10 volte

student_id = input("Enter student ID: ")

OUTPUT_FILE = f"{student_id}_exp2.csv"

# -------------------------
# LOAD IMAGES
# -------------------------

images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Number of images:", len(images))

# Ogni immagine ripetuta N_REPEATS volte
presentation_order = images * N_REPEATS

# Randomize presentation order
random.shuffle(presentation_order)

# Dictionary for storing ratings
ratings = {filename: [] for filename in images}

# -------------------------
# EXPERIMENT
# -------------------------

current_index = 0


def save_rating(rating):
    global current_index

    filename = presentation_order[current_index]

    ratings[filename].append(rating)

    current_index += 1

    if current_index < len(presentation_order):
        show_next_image()
    else:
        finish_experiment()


def show_next_image():

    filename = presentation_order[current_index]

    image_path = os.path.join(IMAGE_FOLDER, filename)

    img = Image.open(image_path)

    img.thumbnail((500, 500))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    progress_label.config(
        text=f"Image {current_index + 1} of {len(presentation_order)}"
    )


def finish_experiment():

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        header = ["filename"] + [f"rating{i+1}" for i in range(N_REPEATS)]
        writer.writerow(header)

        for filename in images:
            writer.writerow([filename] + ratings[filename])

    print("Experiment finished!")
    print("Data saved to:", OUTPUT_FILE)

    root.destroy()


# -------------------------
# GUI
# -------------------------

root = tk.Tk()

root.title("Face Rating Experiment - Part 2")

instruction_label = tk.Label(
    root,
    text="How happy does this person look?\n1 = Very unhappy    5 = Very happy",
    font=("Arial", 18)
)

instruction_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

for rating in range(1, 6):

    button = tk.Button(
        button_frame,
        text=str(rating),
        font=("Arial", 18),
        width=4,
        command=lambda r=rating: save_rating(r)
    )

    button.pack(side=tk.LEFT, padx=5)

progress_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)

progress_label.pack(pady=10)

show_next_image()

root.mainloop()
