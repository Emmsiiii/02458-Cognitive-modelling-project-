# -*- coding: utf-8 -*-
"""CMlab_PCA.ipynb
Original file is located at
    https://colab.research.google.com/drive/1De_1h2vLyMJI99j45lJdHE9QYq9940pI
"""

# ----------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io, color, img_as_float
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------

IMAGES_DIR = "/sample_data/images_grayscale"
IMAGE_EXT = "*.jpg.chip.jpg"

CONVERT_TO_GRAYSCALE = True

N_PC_TO_SHOW = 5
VARIANCE_THRESHOLD = 0.90


# ----------------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------------

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


if IN_COLAB:

    import zipfile

    with zipfile.ZipFile(
        "/content/sample_data/images_grayscale.zip",
        "r"
    ) as z:
        z.extractall(
            "/content/sample_data/images_grayscale"
        )

    IMAGES_PATH = Path(
        "/content/sample_data/images_grayscale"
    )

    BASE_DIR = Path("/content")

else:

    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

    IMAGES_PATH = (
        BASE_DIR
        / "images_grayscale"
    )


# ======================================================================
# PART 3: PCA AND DIMENSION REDUCTION
# ======================================================================


# ----------------------------------------------------------------------
# 1. LOAD IMAGES
# ----------------------------------------------------------------------

def load_images(
    images_path: Path,
    ext: str,
    grayscale: bool
):

    filepaths = sorted(
        images_path.glob(ext)
    )

    if not filepaths:

        raise FileNotFoundError(
            f"No images found in {images_path} "
            f"with extension {ext}."
        )


    images = []


    for fp in filepaths:

        img = img_as_float(
            io.imread(fp)
        )


        if grayscale and img.ndim == 3:

            img = color.rgb2gray(
                img
            )


        images.append(
            img
        )


    shapes = {
        img.shape
        for img in images
    }


    if len(shapes) > 1:

        raise ValueError(
            f"The images have different shapes: {shapes}. "
            "All images must have the same dimensions."
        )


    img_shape = images[0].shape


    # Flatten each image into a vector
    X = np.array(
        [
            img.flatten()
            for img in images
        ]
    )


    return (
        X,
        img_shape,
        filepaths
    )


print(
    "Loading images..."
)


X, img_shape, filepaths = load_images(
    IMAGES_PATH,
    IMAGE_EXT,
    CONVERT_TO_GRAYSCALE
)


n, p = X.shape


print(
    f"Loaded {n} images, each with shape "
    f"{img_shape} -> vectors of dimension {p}"
)


# ----------------------------------------------------------------------
# 2. PCA
# ----------------------------------------------------------------------

# Calculate average image
mean_image = X.mean(
    axis=0
)


# Subtract average image
# IMPORTANT:
# Do NOT divide by standard deviation
X_centered = (
    X
    - mean_image
)


# Run PCA
pca = PCA()


# Scores = representation of images in PCA space
scores = pca.fit_transform(
    X_centered
)


# Principal component vectors
components = pca.components_


# Explained variance for each PC
explained_var_ratio = (
    pca.explained_variance_ratio_
)


# Cumulative explained variance
cum_var = np.cumsum(
    explained_var_ratio
)


# Select number of PCs needed to explain at least 90%
n_components_selected = int(
    np.searchsorted(
        cum_var,
        VARIANCE_THRESHOLD
    )
    + 1
)


print(
    f"Number of PCs required to explain at least "
    f"{VARIANCE_THRESHOLD:.0%} of the variance: "
    f"{n_components_selected} "
    f"(out of a maximum of "
    f"{len(explained_var_ratio)})"
)


# ----------------------------------------------------------------------
# 3. VISUALISE PCA COMPONENTS
# ----------------------------------------------------------------------

def show_pc_effect(
    pc_idx,
    scores,
    components,
    mean_image,
    img_shape,
    grayscale
):

    # Minimum and maximum score
    s_min = (
        scores[:, pc_idx].min()
    )

    s_max = (
        scores[:, pc_idx].max()
    )


    # PC vector
    v = components[
        pc_idx
    ]


    # Mean + minimum score * PC
    img_min = (
        mean_image
        + s_min * v
    ).reshape(
        img_shape
    )


    # Mean face
    img_avg = (
        mean_image.reshape(
            img_shape
        )
    )


    # Mean + maximum score * PC
    img_max = (
        mean_image
        + s_max * v
    ).reshape(
        img_shape
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9, 3)
    )


    cmap = (
        "gray"
        if grayscale
        else None
    )


    for ax, im, title in zip(
        axes,
        [
            img_min,
            img_avg,
            img_max
        ],
        [
            "min score",
            "mean",
            "max score"
        ]
    ):

        ax.imshow(
            np.clip(
                im,
                0,
                1
            ),
            cmap=cmap
        )

        ax.set_title(
            title
        )

        ax.axis(
            "off"
        )


    fig.suptitle(
        f"PC {pc_idx + 1} "
        f"(explained variance: "
        f"{explained_var_ratio[pc_idx]:.2%})"
    )


    fig.tight_layout()


    return fig


print(
    "Generating figures for the first principal components..."
)


for i in range(
    N_PC_TO_SHOW
):

    fig = show_pc_effect(
        i,
        scores,
        components,
        mean_image,
        img_shape,
        CONVERT_TO_GRAYSCALE
    )


    fig.savefig(
        BASE_DIR
        / f"pc_{i + 1}_effect.png",
        dpi=150,
        bbox_inches="tight"
    )


plt.show()


# ----------------------------------------------------------------------
# 4. BAR PLOT OF EXPLAINED VARIANCE
# ----------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 4)
)


ax.bar(
    range(
        1,
        len(explained_var_ratio) + 1
    ),
    explained_var_ratio
)


ax.set_xlabel(
    "Principal component"
)


ax.set_ylabel(
    "Proportion of explained variance"
)


ax.set_title(
    "Scree plot"
)


fig.tight_layout()


fig.savefig(
    BASE_DIR
    / "scree_plot.png",
    dpi=150
)


plt.show()


# ----------------------------------------------------------------------
# 5. CUMULATIVE EXPLAINED VARIANCE
# ----------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 4)
)


ax.plot(
    range(
        1,
        len(cum_var) + 1
    ),
    cum_var,
    marker="o",
    markersize=3
)


ax.axhline(
    VARIANCE_THRESHOLD,
    color="red",
    linestyle="--",
    label=f"threshold {VARIANCE_THRESHOLD:.0%}"
)


ax.axvline(
    n_components_selected,
    color="gray",
    linestyle=":"
)


ax.set_xlabel(
    "Number of components"
)


ax.set_ylabel(
    "Cumulative explained variance"
)


ax.set_title(
    "Cumulative explained variance"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    BASE_DIR
    / "cumulative_variance_plot.png",
    dpi=150
)


plt.show()


# ----------------------------------------------------------------------
# 6. SELECT PCA SCORES FOR REGRESSION
# ----------------------------------------------------------------------

encoding_predictors = (
    scores[
        :,
        :n_components_selected
    ]
)


np.save(
    BASE_DIR
    / "pca_scores_selected.npy",
    encoding_predictors
)


print(
    f"Saved {encoding_predictors.shape[1]} PCA predictors "
    f"for {n} images in pca_scores_selected.npy"
)


# ======================================================================
# PART 4: LINEAR REGRESSION WITH FORWARD SELECTION
# ======================================================================


# ----------------------------------------------------------------------
# 7. LOAD RATINGS
# ----------------------------------------------------------------------

X_reg = (
    encoding_predictors
)


RATINGS_FILE = (
    BASE_DIR
    / "226625.csv"
)


ratings_df = pd.read_csv(
    RATINGS_FILE
)


required_columns = {
    "filename",
    "rating1",
    "rating2"
}


if not required_columns.issubset(
    ratings_df.columns
):

    raise ValueError(
        "The ratings CSV must contain the columns "
        "'filename', 'rating1', and 'rating2'."
    )


# ----------------------------------------------------------------------
# 8. NORMALISE RATINGS IF NECESSARY
# ----------------------------------------------------------------------

all_ratings = ratings_df[
    [
        "rating1",
        "rating2"
    ]
].to_numpy(
    dtype=float
)


rating_min = np.min(
    all_ratings
)

rating_max = np.max(
    all_ratings
)


print(
    "\nOriginal rating range:"
)

print(
    f"Minimum rating: {rating_min}"
)

print(
    f"Maximum rating: {rating_max}"
)


if (
    rating_min > 1
    or rating_max < 5
):

    print(
        "The participant did not use the full 1-5 scale. "
        "Applying min-max normalisation."
    )


    if rating_max == rating_min:

        raise ValueError(
            "All ratings are identical."
        )


    ratings_df[
        "rating1_normalised"
    ] = (
        1
        + 4
        * (
            ratings_df["rating1"]
            - rating_min
        )
        / (
            rating_max
            - rating_min
        )
    )


    ratings_df[
        "rating2_normalised"
    ] = (
        1
        + 4
        * (
            ratings_df["rating2"]
            - rating_min
        )
        / (
            rating_max
            - rating_min
        )
    )


else:

    print(
        "The participant used the full 1-5 scale. "
        "No normalisation is necessary."
    )


    ratings_df[
        "rating1_normalised"
    ] = ratings_df[
        "rating1"
    ].astype(
        float
    )


    ratings_df[
        "rating2_normalised"
    ] = ratings_df[
        "rating2"
    ].astype(
        float
    )


# ----------------------------------------------------------------------
# 9. AVERAGE THE TWO RATINGS FOR EACH IMAGE
# ----------------------------------------------------------------------

ratings_df[
    "mean_rating"
] = ratings_df[
    [
        "rating1_normalised",
        "rating2_normalised"
    ]
].mean(
    axis=1
)


# ----------------------------------------------------------------------
# 10. MATCH RATINGS TO IMAGE ORDER
# ----------------------------------------------------------------------

rating_lookup = (
    ratings_df
    .set_index(
        "filename"
    )[
        "mean_rating"
    ]
)


missing_ratings = [
    fp.name
    for fp in filepaths
    if fp.name not in rating_lookup.index
]


if missing_ratings:

    raise ValueError(
        f"Ratings are missing for "
        f"{len(missing_ratings)} images. "
        f"Example: {missing_ratings[:5]}"
    )


y = np.array(
    [
        rating_lookup.loc[
            fp.name
        ]
        for fp in filepaths
    ],
    dtype=float
)


print(
    "\nRegression data:"
)

print(
    "Number of images:",
    X_reg.shape[0]
)

print(
    "Number of candidate PCs:",
    X_reg.shape[1]
)

print(
    "Number of ratings:",
    len(y)
)


if len(y) != X_reg.shape[0]:

    raise ValueError(
        "The number of ratings must equal "
        "the number of images."
    )


# ----------------------------------------------------------------------
# 11. FORWARD SELECTION WITH CROSS-VALIDATION
# ----------------------------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


def forward_selection_cv(
    X,
    y,
    cv,
    min_improvement=1e-6
):

    selected_features = []

    remaining_features = list(
        range(
            X.shape[1]
        )
    )


    best_mse = np.inf

    history = []


    while remaining_features:

        candidate_results = []


        for feature in remaining_features:

            candidate_features = (
                selected_features
                + [feature]
            )


            model = LinearRegression()


            scores_cv = cross_val_score(
                model,
                X[
                    :,
                    candidate_features
                ],
                y,
                cv=cv,
                scoring="neg_mean_squared_error"
            )


            mse = (
                -scores_cv.mean()
            )


            candidate_results.append(
                (
                    mse,
                    feature
                )
            )


        candidate_results.sort(
            key=lambda x: x[0]
        )


        new_mse, best_feature = (
            candidate_results[0]
        )


        if (
            best_mse
            - new_mse
            > min_improvement
        ):

            selected_features.append(
                best_feature
            )


            remaining_features.remove(
                best_feature
            )


            best_mse = new_mse


            history.append(
                best_mse
            )


            print(
                f"Added PC {best_feature + 1}: "
                f"CV MSE = {best_mse:.4f}"
            )


        else:

            print(
                "\nStopping forward selection: "
                "adding another PC does not improve "
                "cross-validated prediction error."
            )

            break


    return (
        selected_features,
        history
    )


selected_indices, selection_history = (
    forward_selection_cv(
        X_reg,
        y,
        cv
    )
)


selected_indices = np.array(
    selected_indices,
    dtype=int
)


if len(
    selected_indices
) == 0:

    raise RuntimeError(
        "Forward selection did not select any PCs."
    )


print(
    "\nSelected principal components:"
)


print(
    selected_indices
    + 1
)


# ----------------------------------------------------------------------
# 12. FIT FINAL LINEAR REGRESSION
# ----------------------------------------------------------------------

X_selected = X_reg[
    :,
    selected_indices
]


final_model = (
    LinearRegression()
)


final_model.fit(
    X_selected,
    y
)


print(
    "\nFinal linear regression model"
)


print(
    "Selected PCs:",
    selected_indices + 1
)


print(
    "Intercept:",
    final_model.intercept_
)


print(
    "Regression coefficients:",
    final_model.coef_
)


# ----------------------------------------------------------------------
# 13. PREDICT RATINGS FOR ORIGINAL IMAGES
# ----------------------------------------------------------------------

predicted_ratings = (
    final_model.predict(
        X_selected
    )
)


print(
    "\nMinimum predicted rating:",
    predicted_ratings.min()
)


print(
    "Maximum predicted rating:",
    predicted_ratings.max()
)


# ----------------------------------------------------------------------
# 14. VISUALISE SELECTED PCs
# ----------------------------------------------------------------------

print(
    "\nVisualising PCs selected by forward selection..."
)


N_SELECTED_PC_TO_SHOW = min(
    5,
    len(selected_indices)
)


for pc_idx in selected_indices[
    :N_SELECTED_PC_TO_SHOW
]:

    fig = show_pc_effect(
        pc_idx,
        scores,
        components,
        mean_image,
        img_shape,
        CONVERT_TO_GRAYSCALE
    )


    fig.savefig(
        BASE_DIR
        / f"selected_pc_{pc_idx + 1}_effect.png",
        dpi=150,
        bbox_inches="tight"
    )


plt.show()


# ----------------------------------------------------------------------
# 15. FORWARD SELECTION PERFORMANCE
# ----------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7, 4)
)


ax.plot(
    range(
        1,
        len(selection_history) + 1
    ),
    selection_history,
    marker="o"
)


ax.set_xlabel(
    "Number of selected PCs"
)


ax.set_ylabel(
    "Cross-validated mean squared error"
)


ax.set_title(
    "Forward selection"
)


fig.tight_layout()


fig.savefig(
    BASE_DIR
    / "forward_selection_cv.png",
    dpi=150
)


plt.show()


# ======================================================================
# PART 5: GENERATE SYNTHETIC IMAGES
# ======================================================================


# ----------------------------------------------------------------------
# 16. TARGET RATINGS
# ----------------------------------------------------------------------

# Required ratings:
#
# 0.5
# 1.0
# 1.5
# 2.0
# 2.5
# 3.0
# 3.5
# 4.0
# 4.5
# 5.0
# 5.5

target_ratings = np.arange(
    0.5,
    5.5 + 0.5,
    0.5
)


print(
    "\nGenerating synthetic faces..."
)


print(
    "Target ratings:",
    target_ratings
)


# ----------------------------------------------------------------------
# 17. REGRESSION PARAMETERS
# ----------------------------------------------------------------------

w = (
    final_model.coef_
)


delta = (
    final_model.intercept_
)


w_norm_squared = np.dot(
    w,
    w
)


if np.isclose(
    w_norm_squared,
    0
):

    raise ValueError(
        "The regression weight vector has zero length. "
        "Synthetic images cannot be generated."
    )


print(
    "\nRegression intercept:",
    delta
)


print(
    "Squared norm of weight vector:",
    w_norm_squared
)


# ----------------------------------------------------------------------
# 18. FUNCTION TO GENERATE A SYNTHETIC FACE
# ----------------------------------------------------------------------

def generate_synthetic_face(
    target_rating,
    regression_weights,
    intercept,
    selected_indices,
    components,
    mean_image,
    img_shape
):

    """
    Generate a synthetic face for a desired rating.

    Equation:

        scores =
            (target_rating - intercept)
            * w / ||w||^2

    where w contains the regression coefficients
    for the PCs selected using forward selection.
    """


    # Calculate scores in the selected PCA space
    selected_scores = (
        (
            target_rating
            - intercept
        )
        * regression_weights
        / np.dot(
            regression_weights,
            regression_weights
        )
    )


    # Full PCA score vector
    full_scores = np.zeros(
        components.shape[0]
    )


    # Insert scores only for selected PCs
    full_scores[
        selected_indices
    ] = selected_scores


    # Transform from PCA space back to pixel space
    synthetic_vector = (
        mean_image
        + full_scores
        @ components
    )


    # Reshape vector to image
    synthetic_image = (
        synthetic_vector.reshape(
            img_shape
        )
    )


    return (
        synthetic_image,
        selected_scores
    )


# ----------------------------------------------------------------------
# 19. GENERATE ALL 11 SYNTHETIC FACES
# ----------------------------------------------------------------------

synthetic_images = []

synthetic_scores = []


for rating in target_ratings:

    synthetic_image, pc_scores = (
        generate_synthetic_face(
            target_rating=rating,
            regression_weights=w,
            intercept=delta,
            selected_indices=selected_indices,
            components=components,
            mean_image=mean_image,
            img_shape=img_shape
        )
    )


    synthetic_images.append(
        synthetic_image
    )


    synthetic_scores.append(
        pc_scores
    )


synthetic_images = np.array(
    synthetic_images
)


synthetic_scores = np.array(
    synthetic_scores
)


# ----------------------------------------------------------------------
# 20. CHECK THAT SYNTHETIC SCORES GIVE THE CORRECT RATINGS
# ----------------------------------------------------------------------

print(
    "\nChecking synthetic ratings:"
)


checked_predictions = []


for target_rating, pc_scores in zip(
    target_ratings,
    synthetic_scores
):

    predicted_rating = (
        final_model.predict(
            np.asarray(
                pc_scores
            ).reshape(
                1,
                -1
            )
        )[0]
    )


    checked_predictions.append(
        predicted_rating
    )


    print(
        f"Target rating: {target_rating:.1f} "
        f"-> predicted rating: "
        f"{predicted_rating:.4f}"
    )


checked_predictions = np.array(
    checked_predictions
)


if not np.allclose(
    checked_predictions,
    target_ratings,
    atol=1e-10
):

    raise ValueError(
        "The synthetic PCA scores do not reproduce "
        "the requested ratings correctly."
    )


print(
    "\nCheck passed: all synthetic faces reproduce "
    "their requested ratings."
)


# ----------------------------------------------------------------------
# 21. CHECK TARGETS AGAINST TRAINING PREDICTION RANGE
# ----------------------------------------------------------------------

training_prediction_min = (
    predicted_ratings.min()
)

training_prediction_max = (
    predicted_ratings.max()
)


print(
    "\nPrediction range for original images:"
)


print(
    f"{training_prediction_min:.4f} "
    f"to {training_prediction_max:.4f}"
)


print(
    "\nSynthetic targets outside this range:"
)


outside_range = (
    target_ratings[
        (
            target_ratings
            < training_prediction_min
        )
        |
        (
            target_ratings
            > training_prediction_max
        )
    ]
)


if len(
    outside_range
) == 0:

    print(
        "None"
    )

else:

    print(
        outside_range
    )


# ----------------------------------------------------------------------
# 22. DISPLAY ALL 11 SYNTHETIC FACES SIDE-BY-SIDE
# ----------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(target_ratings),
    figsize=(22, 3)
)


for ax, image, rating in zip(
    axes,
    synthetic_images,
    target_ratings
):

    ax.imshow(
        np.clip(
            image,
            0,
            1
        ),
        cmap=(
            "gray"
            if CONVERT_TO_GRAYSCALE
            else None
        )
    )


    ax.set_title(
        f"{rating:.1f}"
    )


    ax.axis(
        "off"
    )


fig.suptitle(
    "Synthetic faces for different predicted ratings"
)


fig.tight_layout()


fig.savefig(
    BASE_DIR
    / "synthetic_faces_all_ratings.png",
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ----------------------------------------------------------------------
# 23. SAVE EACH SYNTHETIC FACE SEPARATELY
# ----------------------------------------------------------------------
for image, rating in zip(
    synthetic_images,
    target_ratings
):

    # Clip values to valid image range
    image_to_save = np.clip(
        image,
        0,
        1
    )

    # Convert from float [0, 1] to uint8 [0, 255]
    image_to_save = (
        image_to_save * 255
    ).astype(np.uint8)

    filename = (
        BASE_DIR
        / f"synthetic_face_rating_{rating:.1f}.png"
    )

    io.imsave(
        filename,
        image_to_save
    )

    # ======================================================================
# PART 6: CHECK WHETHER SYNTHETIC IMAGES EXTEND BEYOND THE DATA
# ======================================================================


# ----------------------------------------------------------------------
# 24. PREDICT RATINGS FOR ALL ORIGINAL INPUT IMAGES
# ----------------------------------------------------------------------

# X_selected contains the selected PC scores for all original images.
input_predicted_ratings = final_model.predict(
    X_selected
)

prediction_min = input_predicted_ratings.min()
prediction_max = input_predicted_ratings.max()


print("\n" + "=" * 60)
print("PART 6: CHECK SYNTHETIC IMAGE RANGE")
print("=" * 60)

print(
    f"\nMinimum predicted rating for input images: "
    f"{prediction_min:.4f}"
)

print(
    f"Maximum predicted rating for input images: "
    f"{prediction_max:.4f}"
)


# ----------------------------------------------------------------------
# 25. COMPARE WITH THE SYNTHETIC TARGET RANGE
# ----------------------------------------------------------------------

synthetic_min = target_ratings.min()
synthetic_max = target_ratings.max()


print(
    f"\nOriginal synthetic rating range: "
    f"{synthetic_min:.4f} to {synthetic_max:.4f}"
)


outside_mask = (
    (target_ratings < prediction_min)
    |
    (target_ratings > prediction_max)
)


outside_targets = target_ratings[
    outside_mask
]


if len(outside_targets) == 0:

    print(
        "\nAll synthetic target ratings are within "
        "the distribution of predicted ratings."
    )

    print(
        "No new synthetic images are necessary."
    )

else:

    print(
        "\nSynthetic target ratings outside "
        "the predicted range:"
    )

    print(
        outside_targets
    )

    print(
        "\nNew synthetic images will be generated "
        "within the predicted rating range."
    )


# ----------------------------------------------------------------------
# 26. INSPECT THE NORM OF THE REGRESSION WEIGHT VECTOR
# ----------------------------------------------------------------------

weight_norm = np.linalg.norm(
    final_model.coef_
)


print(
    f"\nNorm of regression weight vector ||w||: "
    f"{weight_norm:.6f}"
)


# ----------------------------------------------------------------------
# 27. GENERATE A NEW SET INSIDE THE PREDICTED DISTRIBUTION
# ----------------------------------------------------------------------

# Generate 11 equally spaced target ratings from the minimum
# to the maximum rating predicted for the original images.
#
# This guarantees that all generated ratings lie within
# the model's observed prediction range.

safe_target_ratings = np.linspace(
    prediction_min,
    prediction_max,
    11
)


print(
    "\nNew target ratings within predicted distribution:"
)

print(
    np.round(
        safe_target_ratings,
        4
    )
)


safe_synthetic_images = []
safe_synthetic_scores = []


for rating in safe_target_ratings:

    synthetic_image, pc_scores = generate_synthetic_face(
        target_rating=rating,
        regression_weights=w,
        intercept=delta,
        selected_indices=selected_indices,
        components=components,
        mean_image=mean_image,
        img_shape=img_shape
    )

    safe_synthetic_images.append(
        synthetic_image
    )

    safe_synthetic_scores.append(
        pc_scores
    )


safe_synthetic_images = np.array(
    safe_synthetic_images
)

safe_synthetic_scores = np.array(
    safe_synthetic_scores
)


# ----------------------------------------------------------------------
# 28. CHECK THE NEW SYNTHETIC RATINGS
# ----------------------------------------------------------------------

print(
    "\nChecking new synthetic ratings:"
)


safe_checked_predictions = []


for target_rating, pc_scores in zip(
    safe_target_ratings,
    safe_synthetic_scores
):

    predicted_rating = final_model.predict(
        pc_scores.reshape(1, -1)
    )[0]

    safe_checked_predictions.append(
        predicted_rating
    )

    print(
        f"Target: {target_rating:.4f} "
        f"-> predicted: {predicted_rating:.4f}"
    )


safe_checked_predictions = np.array(
    safe_checked_predictions
)


if np.allclose(
    safe_checked_predictions,
    safe_target_ratings,
    atol=1e-10
):

    print(
        "\nCheck passed: all new synthetic faces "
        "produce ratings within the predicted distribution."
    )

else:

    raise ValueError(
        "The new synthetic ratings do not match "
        "their requested target ratings."
    )


# ----------------------------------------------------------------------
# 29. DISPLAY THE NEW SYNTHETIC FACES SIDE-BY-SIDE
# ----------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(safe_target_ratings),
    figsize=(22, 3)
)


for ax, image, rating in zip(
    axes,
    safe_synthetic_images,
    safe_target_ratings
):

    ax.imshow(
        np.clip(
            image,
            0,
            1
        ),
        cmap=(
            "gray"
            if CONVERT_TO_GRAYSCALE
            else None
        )
    )

    ax.set_title(
        f"{rating:.2f}"
    )

    ax.axis(
        "off"
    )


fig.suptitle(
    "Synthetic faces within the predicted rating distribution"
)


fig.tight_layout()


fig.savefig(
    BASE_DIR / "synthetic_faces_within_distribution.png",
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ----------------------------------------------------------------------
# 30. SAVE THE NEW SYNTHETIC FACES SEPARATELY
# ----------------------------------------------------------------------

for image, rating in zip(
    safe_synthetic_images,
    safe_target_ratings
):

    # Convert float image [0, 1] to uint8 [0, 255]
    image_to_save = (
        np.clip(
            image,
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )


    filename = (
        BASE_DIR
        / f"safe_synthetic_face_rating_{rating:.2f}.png"
    )


    io.imsave(
        filename,
        image_to_save
    )


print(
    "\nSaved new synthetic faces within "
    "the predicted rating distribution."
)


print(
    "Saved combined figure as "
    "'synthetic_faces_within_distribution.png'."
)