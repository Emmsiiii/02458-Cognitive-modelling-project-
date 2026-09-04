import os
import random
import csv
import tkinter as tk
from PIL import Image, ImageTk

# -------------------------
# SETTINGS
# -------------------------

IMAGE_FOLDER = "/Users/emmsi/Desktop/images_grayscale"

student_id = input("Enter student ID: ")

OUTPUT_FILE = f"{student_id}.csv"

# -------------------------
# LOAD IMAGES
# -------------------------

images = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Number of images:", len(images))

# Each image should be shown twice
presentation_order = images * 2

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

    # Save rating for this image
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

    # Resize only for display
    img.thumbnail((500, 500))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo

    progress_label.config(
        text=f"Image {current_index + 1} of {len(presentation_order)}"
    )


def finish_experiment():

    # Save CSV
    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "rating1",
            "rating2"
        ])

        for filename in images:

            writer.writerow([
                filename,
                ratings[filename][0],
                ratings[filename][1]
            ])

    print("Experiment finished!")
    print("Data saved to:", OUTPUT_FILE)

    root.destroy()


# -------------------------
# GUI
# -------------------------

root = tk.Tk()

root.title("Face Rating Experiment")

# Instructions
instruction_label = tk.Label(
    root,
    text="How happy does this person look?\n1 = Very unhappy    5 = Very happy",
    font=("Arial", 18)
)

instruction_label.pack(pady=10)

# Image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Rating buttons
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

# Progress
progress_label = tk.Label(
    root,
    text="",
    font=("Arial", 12)
)

progress_label.pack(pady=10)

# Show first image
show_next_image()

root.mainloop()