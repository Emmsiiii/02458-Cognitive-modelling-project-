# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io, color, img_as_float
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


# ======================================================================
# SETTINGS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGES_PATH = BASE_DIR / "images_grayscale"

RATINGS_FILES = [
    BASE_DIR / "226625.csv",
    BASE_DIR / "s262010.csv",
]

CONVERT_TO_GRAYSCALE = True

# Number of first PCs to visualise
N_PC_TO_SHOW = 5

# PCA dimensionality reduction threshold
VARIANCE_THRESHOLD = 0.90

# Cross-validation
N_CV_FOLDS = 5
RANDOM_STATE = 42

# Improved forward-selection stopping criteria.
#
# A new PC is only accepted if it reduces CV MSE by at least:
#   1% of the current MSE
# OR
#   0.005 absolute MSE
#
# The larger requirement is used.
MIN_RELATIVE_IMPROVEMENT = 0.01
MIN_ABSOLUTE_IMPROVEMENT = 0.005


# ======================================================================
# PART 2: RATINGS
# ======================================================================

print("\n" + "=" * 70)
print("PART 2: PARTICIPANT RATINGS")
print("=" * 70)

participant_data = []
participant_summaries = []


for ratings_file in RATINGS_FILES:

    print("\n" + "=" * 70)
    print(f"PARTICIPANT: {ratings_file.stem}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"Could not find ratings file:\n{ratings_file}"
        )

    df = pd.read_csv(ratings_file)

    required_columns = {
        "filename",
        "rating1",
        "rating2",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{ratings_file.name} must contain "
            "'filename', 'rating1', and 'rating2'."
        )

    if df[
        ["filename", "rating1", "rating2"]
    ].isnull().any().any():

        raise ValueError(
            f"{ratings_file.name} contains missing values."
        )

    if df["filename"].duplicated().any():

        duplicates = (
            df.loc[
                df["filename"].duplicated(),
                "filename"
            ]
            .tolist()
        )

        raise ValueError(
            f"{ratings_file.name} contains duplicate filenames.\n"
            f"Examples: {duplicates[:5]}"
        )

    # ------------------------------------------------------------------
    # Original ratings
    # ------------------------------------------------------------------

    all_ratings = df[
        ["rating1", "rating2"]
    ].to_numpy(dtype=float)

    all_ratings_flat = all_ratings.flatten()

    if (
        np.any(all_ratings_flat < 1)
        or np.any(all_ratings_flat > 5)
    ):
        raise ValueError(
            f"{ratings_file.name} contains ratings outside 1-5."
        )

    # ------------------------------------------------------------------
    # Original-rating histogram
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        all_ratings_flat,
        bins=np.arange(0.5, 6.0, 1),
        edgecolor="black",
    )

    ax.set_xticks([1, 2, 3, 4, 5])

    ax.set_xlabel("Happiness rating")
    ax.set_ylabel("Number of responses")

    ax.set_title(
        f"Rating distribution – participant {ratings_file.stem}"
    )

    fig.tight_layout()

    histogram_file = (
        BASE_DIR
        / f"rating_histogram_{ratings_file.stem}.png"
    )

    fig.savefig(
        histogram_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved histogram as: {histogram_file.name}"
    )

    # ------------------------------------------------------------------
    # Rating range
    # ------------------------------------------------------------------

    rating_min = float(
        np.min(all_ratings_flat)
    )

    rating_max = float(
        np.max(all_ratings_flat)
    )

    print(
        f"\nParticipant {ratings_file.stem}"
    )

    print(
        f"Minimum rating: {rating_min}"
    )

    print(
        f"Maximum rating: {rating_max}"
    )

    print("\nRating counts:")

    rating_counts = {}

    for rating in range(1, 6):

        count = int(
            np.sum(
                all_ratings_flat == rating
            )
        )

        rating_counts[rating] = count

        print(
            f"Rating {rating}: {count}"
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    # According to the project instructions:
    # normalise only if participant did not use full 1-5 range.

    if (
        rating_min > 1
        or rating_max < 5
    ):

        print(
            "\nParticipant did NOT use "
            "the full 1-5 rating range."
        )

        print(
            "Applying min-max normalisation."
        )

        if np.isclose(
            rating_min,
            rating_max
        ):
            raise ValueError(
                f"All ratings in {ratings_file.name} are identical."
            )

        for column in [
            "rating1",
            "rating2",
        ]:

            df[
                f"{column}_normalised"
            ] = (
                1
                + 4
                * (
                    df[column]
                    - rating_min
                )
                / (
                    rating_max
                    - rating_min
                )
            )

        was_normalised = True

    else:

        print(
            "\nParticipant used the full 1-5 rating range."
        )

        print(
            "No normalisation necessary."
        )

        df[
            "rating1_normalised"
        ] = df["rating1"].astype(float)

        df[
            "rating2_normalised"
        ] = df["rating2"].astype(float)

        was_normalised = False

    # ------------------------------------------------------------------
    # Normalised histogram, only if needed
    # ------------------------------------------------------------------

    if was_normalised:

        normalised_ratings = df[
            [
                "rating1_normalised",
                "rating2_normalised",
            ]
        ].to_numpy().flatten()

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.hist(
            normalised_ratings,
            bins=np.linspace(
                1,
                5,
                10
            ),
            edgecolor="black",
        )

        ax.set_xlabel(
            "Normalised happiness rating"
        )

        ax.set_ylabel(
            "Number of responses"
        )

        ax.set_title(
            f"Normalised rating distribution – "
            f"participant {ratings_file.stem}"
        )

        fig.tight_layout()

        filename = (
            BASE_DIR
            / (
                f"rating_histogram_"
                f"{ratings_file.stem}_normalised.png"
            )
        )

        fig.savefig(
            filename,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ------------------------------------------------------------------
    # Average first and second presentation
    # ------------------------------------------------------------------

    df[
        "participant_mean_rating"
    ] = df[
        [
            "rating1_normalised",
            "rating2_normalised",
        ]
    ].mean(axis=1)

    participant_data.append(

        df[
            [
                "filename",
                "participant_mean_rating",
            ]
        ].rename(

            columns={
                "participant_mean_rating":
                    ratings_file.stem
            }

        )
    )

    participant_summaries.append(
        {
            "participant": ratings_file.stem,
            "minimum": rating_min,
            "maximum": rating_max,
            "normalised": was_normalised,
            **{
                f"count_{r}":
                    rating_counts[r]
                for r in range(1, 6)
            },
        }
    )


# ======================================================================
# CHECK PARTICIPANTS RATED SAME IMAGES
# ======================================================================

reference_filenames = set(
    participant_data[0]["filename"]
)

for participant_df in participant_data[1:]:

    filenames = set(
        participant_df["filename"]
    )

    if filenames != reference_filenames:

        missing = (
            reference_filenames
            - filenames
        )

        extra = (
            filenames
            - reference_filenames
        )

        raise ValueError(
            "Participant files do not contain the same images.\n"
            f"Missing examples: {list(missing)[:5]}\n"
            f"Extra examples: {list(extra)[:5]}"
        )


# ======================================================================
# COMBINE PARTICIPANTS
# ======================================================================

combined_ratings = participant_data[0].copy()

for participant_df in participant_data[1:]:

    combined_ratings = (
        combined_ratings.merge(
            participant_df,
            on="filename",
            how="inner",
        )
    )


participant_columns = [
    file.stem
    for file in RATINGS_FILES
]


combined_ratings[
    "mean_rating"
] = combined_ratings[
    participant_columns
].mean(axis=1)


print("\n" + "=" * 70)
print("COMBINED RATINGS")
print("=" * 70)

print(
    "Number of images rated by all participants:",
    len(combined_ratings),
)

print(
    "\nMean rating across all images:",
    combined_ratings["mean_rating"].mean(),
)


combined_ratings.to_csv(
    BASE_DIR / "combined_ratings.csv",
    index=False,
)


print(
    "Saved combined ratings as 'combined_ratings.csv'."
)


# ======================================================================
# PART 3: PCA
# ======================================================================

print("\n" + "=" * 70)
print("PART 3: PCA")
print("=" * 70)


# ======================================================================
# LOAD IMAGES
# ======================================================================

def load_images(
    images_path,
    grayscale=True,
):

    if not images_path.exists():

        raise FileNotFoundError(
            f"Image folder does not exist:\n"
            f"{images_path}"
        )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    filepaths = sorted(
        [
            fp
            for fp in images_path.iterdir()
            if (
                fp.is_file()
                and fp.suffix.lower()
                in valid_extensions
            )
        ],
        key=lambda x: x.name,
    )

    if not filepaths:

        raise FileNotFoundError(
            f"No images found in:\n{images_path}"
        )

    images = []

    for fp in filepaths:

        img = img_as_float(
            io.imread(fp)
        )

        if grayscale and img.ndim == 3:

            if img.shape[2] == 4:
                img = img[:, :, :3]

            img = color.rgb2gray(img)

        images.append(img)

    shapes = {
        img.shape
        for img in images
    }

    if len(shapes) != 1:

        raise ValueError(
            "The images do not all have the same dimensions.\n"
            f"Found shapes: {shapes}"
        )

    img_shape = images[0].shape

    X = np.array(
        [
            image.flatten()
            for image in images
        ],
        dtype=float,
    )

    return (
        X,
        img_shape,
        filepaths,
    )


print(
    "\nLoading images from:"
)

print(
    IMAGES_PATH
)


X, img_shape, filepaths = load_images(
    IMAGES_PATH,
    CONVERT_TO_GRAYSCALE,
)


n, p = X.shape


print(
    f"\nLoaded {n} images."
)

print(
    f"Image shape: {img_shape}"
)

print(
    f"Number of pixels/features: {p}"
)


# ======================================================================
# CHECK IMAGE / RATING MATCH
# ======================================================================

image_names = {
    fp.name
    for fp in filepaths
}

rating_names = set(
    combined_ratings["filename"]
)


missing_ratings = (
    image_names
    - rating_names
)

missing_images = (
    rating_names
    - image_names
)


if missing_ratings:

    raise ValueError(
        f"{len(missing_ratings)} images have no ratings.\n"
        f"Examples: {list(missing_ratings)[:5]}"
    )


if missing_images:

    raise ValueError(
        f"{len(missing_images)} rated files are missing "
        "from images_grayscale.\n"
        f"Examples: {list(missing_images)[:5]}"
    )


# ======================================================================
# MEAN-CENTER IMAGES
#
# IMPORTANT:
# subtract the mean image.
# Do NOT divide pixels by standard deviation.
# ======================================================================

mean_image = X.mean(axis=0)

X_centered = (
    X
    - mean_image
)


# ======================================================================
# PCA
# ======================================================================

pca = PCA()

scores = pca.fit_transform(
    X_centered
)

components = (
    pca.components_
)

explained_var_ratio = (
    pca.explained_variance_ratio_
)

cum_var = np.cumsum(
    explained_var_ratio
)


n_components_selected = int(
    np.searchsorted(
        cum_var,
        VARIANCE_THRESHOLD,
    )
    + 1
)


print(
    f"\nPCs required to explain at least "
    f"{VARIANCE_THRESHOLD:.0%} "
    f"of image variance:"
)

print(
    n_components_selected
)


# ======================================================================
# PCA VISUALISATION
# ======================================================================

def show_pc_effect(
    pc_idx,
    scores,
    components,
    mean_image,
    img_shape,
):

    minimum_score = (
        scores[:, pc_idx].min()
    )

    maximum_score = (
        scores[:, pc_idx].max()
    )

    component = (
        components[pc_idx]
    )

    minimum_face = (
        mean_image
        + minimum_score
        * component
    ).reshape(img_shape)

    average_face = (
        mean_image.reshape(
            img_shape
        )
    )

    maximum_face = (
        mean_image
        + maximum_score
        * component
    ).reshape(img_shape)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9, 3),
    )

    images = [
        minimum_face,
        average_face,
        maximum_face,
    ]

    titles = [
        "Minimum score",
        "Mean face",
        "Maximum score",
    ]

    for ax, image, title in zip(
        axes,
        images,
        titles,
    ):

        ax.imshow(
            np.clip(
                image,
                0,
                1,
            ),
            cmap="gray",
        )

        ax.set_title(title)

        ax.axis("off")

    fig.suptitle(
        f"PC {pc_idx + 1} "
        f"(variance = "
        f"{explained_var_ratio[pc_idx]:.2%})"
    )

    fig.tight_layout()

    return fig


print(
    "\nGenerating visualisations of the first PCs..."
)


for i in range(
    min(
        N_PC_TO_SHOW,
        len(components),
    )
):

    fig = show_pc_effect(
        i,
        scores,
        components,
        mean_image,
        img_shape,
    )

    fig.savefig(
        BASE_DIR
        / f"pc_{i + 1}_effect.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ======================================================================
# SCREE PLOT
# ======================================================================

fig, ax = plt.subplots(
    figsize=(9, 4)
)

ax.bar(
    np.arange(
        1,
        len(explained_var_ratio) + 1,
    ),
    explained_var_ratio,
)

ax.set_xlabel(
    "Principal component"
)

ax.set_ylabel(
    "Explained variance ratio"
)

ax.set_title(
    "Explained variance by principal component"
)

fig.tight_layout()

fig.savefig(
    BASE_DIR / "scree_plot.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ======================================================================
# CUMULATIVE VARIANCE PLOT
# ======================================================================

fig, ax = plt.subplots(
    figsize=(8, 4)
)

ax.plot(
    np.arange(
        1,
        len(cum_var) + 1,
    ),
    cum_var,
)

ax.axhline(
    VARIANCE_THRESHOLD,
    linestyle="--",
    label=(
        f"{VARIANCE_THRESHOLD:.0%} threshold"
    ),
)

ax.axvline(
    n_components_selected,
    linestyle=":",
)

ax.set_xlabel(
    "Number of principal components"
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
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ======================================================================
# PCs USED AS CANDIDATES FOR REGRESSION
# ======================================================================

X_reg = scores[
    :,
    :n_components_selected
]


np.save(
    BASE_DIR
    / "pca_scores_selected.npy",
    X_reg,
)


print(
    f"\nUsing {X_reg.shape[1]} PCs "
    "as candidates for regression."
)


# ======================================================================
# PART 4: LINEAR REGRESSION
# ======================================================================

print("\n" + "=" * 70)
print("PART 4: LINEAR REGRESSION")
print("=" * 70)


# ======================================================================
# MATCH RATINGS TO IMAGE ORDER
# ======================================================================

rating_lookup = (
    combined_ratings
    .set_index("filename")[
        "mean_rating"
    ]
)


y = np.array(
    [
        rating_lookup.loc[
            fp.name
        ]
        for fp in filepaths
    ],
    dtype=float,
)


print(
    "\nRegression data:"
)

print(
    "Number of images:",
    len(y),
)

print(
    "Number of candidate PCs:",
    X_reg.shape[1],
)

print(
    "Number of ratings:",
    len(y),
)

print(
    "Mean rating:",
    y.mean(),
)

print(
    "Minimum rating:",
    y.min(),
)

print(
    "Maximum rating:",
    y.max(),
)


# ======================================================================
# CROSS VALIDATION
# ======================================================================

cv = KFold(
    n_splits=N_CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ======================================================================
# IMPROVED FORWARD SELECTION
# ======================================================================

def forward_selection_cv(
    X,
    y,
    cv,
    min_relative_improvement=0.01,
    min_absolute_improvement=0.005,
):

    selected_features = []

    remaining_features = list(
        range(X.shape[1])
    )

    selection_history = []

    # ------------------------------------------------------------------
    # Baseline = intercept-only model
    # ------------------------------------------------------------------

    baseline_mses = []

    for (
        train_index,
        test_index,
    ) in cv.split(X):

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        prediction = np.full(
            len(y_test),
            y_train.mean(),
        )

        mse = np.mean(
            (
                y_test
                - prediction
            )
            ** 2
        )

        baseline_mses.append(
            mse
        )

    current_mse = float(
        np.mean(
            baseline_mses
        )
    )

    baseline_mse = (
        current_mse
    )

    print(
        "\nBaseline CV MSE "
        "(intercept only): "
        f"{baseline_mse:.4f}"
    )

    # ------------------------------------------------------------------
    # Forward-selection loop
    # ------------------------------------------------------------------

    while remaining_features:

        candidate_results = []

        for feature in remaining_features:

            candidate_features = (
                selected_features
                + [feature]
            )

            model = (
                LinearRegression()
            )

            scores_cv = (
                cross_val_score(
                    model,
                    X[
                        :,
                        candidate_features
                    ],
                    y,
                    cv=cv,
                    scoring=(
                        "neg_mean_squared_error"
                    ),
                )
            )

            mse = float(
                -scores_cv.mean()
            )

            candidate_results.append(
                (
                    mse,
                    feature,
                )
            )

        # Lowest CV MSE is best
        candidate_results.sort(
            key=lambda x: x[0]
        )

        candidate_mse = (
            candidate_results[0][0]
        )

        best_feature = (
            candidate_results[0][1]
        )

        absolute_improvement = (
            current_mse
            - candidate_mse
        )

        relative_improvement = (
            absolute_improvement
            / current_mse
        )

        required_improvement = max(
            min_absolute_improvement,
            min_relative_improvement
            * current_mse,
        )

        print(
            "\n----------------------------------------"
        )

        print(
            f"Best candidate: "
            f"PC {best_feature + 1}"
        )

        print(
            f"Current CV MSE: "
            f"{current_mse:.4f}"
        )

        print(
            f"Candidate CV MSE: "
            f"{candidate_mse:.4f}"
        )

        print(
            f"Absolute improvement: "
            f"{absolute_improvement:.4f}"
        )

        print(
            f"Relative improvement: "
            f"{relative_improvement:.2%}"
        )

        print(
            f"Required improvement: "
            f"{required_improvement:.4f}"
        )

        # --------------------------------------------------------------
        # Accept only meaningful improvement
        # --------------------------------------------------------------

        if (
            absolute_improvement
            >= required_improvement
        ):

            selected_features.append(
                best_feature
            )

            remaining_features.remove(
                best_feature
            )

            current_mse = (
                candidate_mse
            )

            selection_history.append(
                current_mse
            )

            print(
                f"Accepted PC "
                f"{best_feature + 1}"
            )

        else:

            print(
                "\nStopping forward selection."
            )

            print(
                "The best remaining PC "
                "does not improve CV MSE enough."
            )

            break

    return (
        selected_features,
        selection_history,
        baseline_mse,
    )


# ======================================================================
# RUN FORWARD SELECTION
# ======================================================================

(
    selected_indices,
    selection_history,
    baseline_cv_mse,
) = forward_selection_cv(

    X_reg,
    y,
    cv,

    min_relative_improvement=(
        MIN_RELATIVE_IMPROVEMENT
    ),

    min_absolute_improvement=(
        MIN_ABSOLUTE_IMPROVEMENT
    ),
)


selected_indices = np.array(
    selected_indices,
    dtype=int,
)


if len(selected_indices) == 0:

    raise RuntimeError(
        "Forward selection selected no PCs."
    )


print(
    "\nSelected principal components:"
)

print(
    selected_indices + 1
)

print(
    "\nNumber of selected PCs:",
    len(selected_indices),
)


# ======================================================================
# FINAL REGRESSION MODEL
# ======================================================================

X_selected = X_reg[
    :,
    selected_indices
]


final_model = LinearRegression()

final_model.fit(
    X_selected,
    y,
)


print(
    "\nFinal linear regression model"
)

print(
    "Selected PCs:",
    selected_indices + 1,
)

print(
    "Intercept:",
    final_model.intercept_,
)

print(
    "Regression coefficients:"
)

print(
    final_model.coef_
)


# ======================================================================
# ORIGINAL-IMAGE PREDICTIONS
# ======================================================================

predicted_ratings = (
    final_model.predict(
        X_selected
    )
)


prediction_min = float(
    predicted_ratings.min()
)

prediction_max = float(
    predicted_ratings.max()
)


print(
    "\nPrediction range for original images:"
)

print(
    f"{prediction_min:.4f} "
    f"to "
    f"{prediction_max:.4f}"
)


# ======================================================================
# VISUALISE SELECTED PCs
# ======================================================================

print(
    "\nVisualising PCs selected "
    "by forward selection..."
)


for pc_idx in selected_indices:

    fig = show_pc_effect(
        pc_idx,
        scores,
        components,
        mean_image,
        img_shape,
    )

    fig.savefig(
        BASE_DIR
        / (
            f"selected_pc_"
            f"{pc_idx + 1}_effect.png"
        ),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ======================================================================
# FORWARD-SELECTION FIGURE
# ======================================================================

fig, ax = plt.subplots(
    figsize=(7, 4)
)


mse_history = (
    [baseline_cv_mse]
    + selection_history
)


ax.plot(
    np.arange(
        len(mse_history)
    ),
    mse_history,
    marker="o",
)


ax.set_xlabel(
    "Number of selected PCs"
)

ax.set_ylabel(
    "Cross-validated MSE"
)

ax.set_title(
    "Forward selection performance"
)

ax.set_xticks(
    np.arange(
        len(mse_history)
    )
)

fig.tight_layout()

fig.savefig(
    BASE_DIR
    / "forward_selection_cv.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ======================================================================
# SAVE MODEL PREDICTIONS
# ======================================================================

model_predictions = pd.DataFrame(
    {
        "filename": [
            fp.name
            for fp in filepaths
        ],

        "observed_rating":
            y,

        "predicted_rating":
            predicted_ratings,
    }
)


model_predictions.to_csv(
    BASE_DIR
    / "model_predictions.csv",
    index=False,
)


# ======================================================================
# PART 5: SYNTHETIC FACES
# ======================================================================

print("\n" + "=" * 70)
print("PART 5: SYNTHETIC IMAGES")
print("=" * 70)


target_ratings = np.arange(
    0.5,
    5.5 + 0.5,
    0.5,
)


print(
    "\nRequested target ratings:"
)

print(
    target_ratings
)


# ======================================================================
# REGRESSION PARAMETERS
# ======================================================================

w = np.asarray(
    final_model.coef_,
    dtype=float,
)

intercept = float(
    final_model.intercept_
)

w_norm_squared = float(
    np.dot(
        w,
        w
    )
)

weight_norm = float(
    np.linalg.norm(
        w
    )
)


print(
    "\nRegression intercept:",
    intercept,
)

print(
    "Norm of regression weight vector ||w||:",
    weight_norm,
)


if np.isclose(
    w_norm_squared,
    0,
):
    raise ValueError(
        "Regression weight vector is "
        "approximately zero. "
        "Synthetic faces cannot be generated."
    )


# ======================================================================
# SYNTHETIC-FACE FUNCTION
# ======================================================================

def generate_synthetic_face(
    target_rating,
    regression_weights,
    intercept,
    selected_indices,
    components,
    mean_image,
    img_shape,
):

    # Minimum-norm solution satisfying:
    #
    # target = intercept + w^T s

    selected_scores = (
        (
            target_rating
            - intercept
        )
        * regression_weights
        / np.dot(
            regression_weights,
            regression_weights,
        )
    )

    # Full vector in PCA space
    full_scores = np.zeros(
        components.shape[0],
        dtype=float,
    )

    full_scores[
        selected_indices
    ] = selected_scores

    # Reconstruct image
    synthetic_vector = (
        mean_image
        + full_scores
        @ components
    )

    synthetic_image = (
        synthetic_vector.reshape(
            img_shape
        )
    )

    return (
        synthetic_image,
        selected_scores,
    )


# ======================================================================
# GENERATE ORIGINAL 0.5 - 5.5 SYNTHETIC CONTINUUM
# ======================================================================

synthetic_images = []
synthetic_selected_scores = []


for target in target_ratings:

    image, selected_scores = (
        generate_synthetic_face(

            target_rating=target,

            regression_weights=w,

            intercept=intercept,

            selected_indices=(
                selected_indices
            ),

            components=components,

            mean_image=mean_image,

            img_shape=img_shape,
        )
    )

    synthetic_images.append(
        image
    )

    synthetic_selected_scores.append(
        selected_scores
    )


synthetic_images = np.array(
    synthetic_images
)

synthetic_selected_scores = np.array(
    synthetic_selected_scores
)


# ======================================================================
# VERIFY SYNTHETIC RATINGS
# ======================================================================

print(
    "\nChecking synthetic ratings:"
)


synthetic_predictions = []


for (
    target,
    selected_scores,
) in zip(
    target_ratings,
    synthetic_selected_scores,
):

    predicted = (
        final_model.predict(
            selected_scores.reshape(
                1,
                -1
            )
        )[0]
    )

    synthetic_predictions.append(
        predicted
    )

    print(
        f"Target: {target:.1f} "
        f"-> predicted: "
        f"{predicted:.4f}"
    )


synthetic_predictions = np.array(
    synthetic_predictions
)


if not np.allclose(
    synthetic_predictions,
    target_ratings,
    atol=1e-8,
):

    raise ValueError(
        "Synthetic ratings do not "
        "match requested targets."
    )


print(
    "\nCheck passed."
)


# ======================================================================
# SAVE ORIGINAL SYNTHETIC CONTINUUM
# ======================================================================

fig, axes = plt.subplots(
    1,
    len(target_ratings),
    figsize=(22, 3),
)


for (
    ax,
    image,
    target,
) in zip(
    axes,
    synthetic_images,
    target_ratings,
):

    ax.imshow(
        np.clip(
            image,
            0,
            1,
        ),
        cmap="gray",
    )

    ax.set_title(
        f"{target:.1f}"
    )

    ax.axis("off")


fig.suptitle(
    "Synthetic faces for predicted happiness ratings"
)

fig.tight_layout()

fig.savefig(
    BASE_DIR
    / "synthetic_faces_all_ratings.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


for (
    image,
    target,
) in zip(
    synthetic_images,
    target_ratings,
):

    image_uint8 = (
        np.clip(
            image,
            0,
            1,
        )
        * 255
    ).astype(np.uint8)

    io.imsave(
        BASE_DIR
        / (
            f"synthetic_face_rating_"
            f"{target:.1f}.png"
        ),
        image_uint8,
    )


# ======================================================================
# PART 6: CHECK RANGE
# ======================================================================

print("\n" + "=" * 70)
print("PART 6: CHECK SYNTHETIC IMAGE RANGE")
print("=" * 70)


print(
    "\nMinimum predicted rating "
    "for original input images:"
)

print(
    f"{prediction_min:.4f}"
)


print(
    "\nMaximum predicted rating "
    "for original input images:"
)

print(
    f"{prediction_max:.4f}"
)


print(
    "\nRequested synthetic rating range:"
)

print(
    f"{target_ratings.min():.1f} "
    f"to "
    f"{target_ratings.max():.1f}"
)


# ======================================================================
# CHECK TARGETS OUTSIDE TRAINING-PREDICTION DISTRIBUTION
# ======================================================================

outside_mask = (
    (target_ratings < prediction_min)
    |
    (target_ratings > prediction_max)
)


outside_targets = (
    target_ratings[
        outside_mask
    ]
)


safe_target_ratings = None


if len(outside_targets) == 0:

    print(
        "\nAll requested synthetic ratings "
        "are within the prediction range."
    )

    print(
        "No replacement synthetic "
        "faces are necessary."
    )


else:

    print(
        "\nSynthetic target ratings "
        "outside the prediction range:"
    )

    print(
        outside_targets
    )

    print(
        "\nThe original 0.5-5.5 continuum "
        "extends beyond the distribution "
        "of predictions for the input images."
    )

    print(
        "Generating a new set of "
        "11 synthetic faces inside "
        "the prediction range."
    )

    # ------------------------------------------------------------------
    # 11 equally spaced safe targets
    # ------------------------------------------------------------------

    safe_target_ratings = np.linspace(
        prediction_min,
        prediction_max,
        11,
    )


    print(
        "\nNew target ratings:"
    )

    print(
        safe_target_ratings
    )


    safe_images = []

    safe_scores = []


    for target in safe_target_ratings:

        image, selected_scores = (
            generate_synthetic_face(

                target_rating=target,

                regression_weights=w,

                intercept=intercept,

                selected_indices=(
                    selected_indices
                ),

                components=components,

                mean_image=mean_image,

                img_shape=img_shape,
            )
        )

        safe_images.append(
            image
        )

        safe_scores.append(
            selected_scores
        )


    safe_images = np.array(
        safe_images
    )

    safe_scores = np.array(
        safe_scores
    )


    # ------------------------------------------------------------------
    # Verify safe synthetic ratings
    # ------------------------------------------------------------------

    print(
        "\nChecking new synthetic ratings:"
    )


    safe_predictions = []


    for (
        target,
        selected_scores,
    ) in zip(
        safe_target_ratings,
        safe_scores,
    ):

        predicted = (
            final_model.predict(
                selected_scores.reshape(
                    1,
                    -1
                )
            )[0]
        )

        safe_predictions.append(
            predicted
        )

        print(
            f"Target: {target:.4f} "
            f"-> predicted: "
            f"{predicted:.4f}"
        )


    safe_predictions = np.array(
        safe_predictions
    )


    if not np.allclose(
        safe_predictions,
        safe_target_ratings,
        atol=1e-8,
    ):

        raise ValueError(
            "Replacement synthetic ratings "
            "do not match the requested values."
        )


    # ------------------------------------------------------------------
    # Plot safe faces
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(safe_target_ratings),
        figsize=(22, 3),
    )


    for (
        ax,
        image,
        target,
    ) in zip(
        axes,
        safe_images,
        safe_target_ratings,
    ):

        ax.imshow(
            np.clip(
                image,
                0,
                1,
            ),
            cmap="gray",
        )

        ax.set_title(
            f"{target:.2f}"
        )

        ax.axis("off")


    fig.suptitle(
        "Synthetic faces within the predicted rating distribution"
    )

    fig.tight_layout()


    fig.savefig(
        BASE_DIR
        / "synthetic_faces_within_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(fig)


    # ------------------------------------------------------------------
    # Save safe faces individually
    # ------------------------------------------------------------------

    for (
        image,
        target,
    ) in zip(
        safe_images,
        safe_target_ratings,
    ):

        image_uint8 = (
            np.clip(
                image,
                0,
                1,
            )
            * 255
        ).astype(np.uint8)

        io.imsave(
            BASE_DIR
            / (
                f"safe_synthetic_face_"
                f"rating_{target:.2f}.png"
            ),
            image_uint8,
        )


    print(
        "\nSaved the replacement synthetic faces."
    )


# ======================================================================
# ANALYSIS SUMMARY FILE
# ======================================================================

summary_file = (
    BASE_DIR
    / "analysis_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "COGNITIVE MODELLING PROJECT\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    # ------------------------------------------------------------------
    # Participants
    # ------------------------------------------------------------------

    file.write(
        "PARTICIPANTS\n"
    )

    file.write(
        "-" * 60
        + "\n"
    )


    for summary in participant_summaries:

        file.write(
            f"Participant: "
            f"{summary['participant']}\n"
        )

        file.write(
            f"Rating range: "
            f"{summary['minimum']:.1f} "
            f"to "
            f"{summary['maximum']:.1f}\n"
        )

        file.write(
            f"Normalised: "
            f"{summary['normalised']}\n"
        )

        for rating in range(1, 6):

            file.write(
                f"Rating {rating}: "
                f"{summary[f'count_{rating}']}\n"
            )

        file.write("\n")


    # ------------------------------------------------------------------
    # Images / PCA
    # ------------------------------------------------------------------

    file.write(
        "PCA\n"
    )

    file.write(
        "-" * 60
        + "\n"
    )

    file.write(
        f"Number of images: {n}\n"
    )

    file.write(
        f"Image shape: {img_shape}\n"
    )

    file.write(
        f"Number of pixel features: {p}\n"
    )

    file.write(
        f"Variance threshold: "
        f"{VARIANCE_THRESHOLD:.0%}\n"
    )

    file.write(
        f"PCs required: "
        f"{n_components_selected}\n\n"
    )


    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    file.write(
        "REGRESSION\n"
    )

    file.write(
        "-" * 60
        + "\n"
    )

    file.write(
        f"Mean observed rating: "
        f"{y.mean():.4f}\n"
    )

    file.write(
        f"Observed rating range: "
        f"{y.min():.4f} "
        f"to "
        f"{y.max():.4f}\n"
    )

    file.write(
        f"Baseline CV MSE: "
        f"{baseline_cv_mse:.4f}\n"
    )

    if selection_history:

        file.write(
            f"Final forward-selection "
            f"CV MSE: "
            f"{selection_history[-1]:.4f}\n"
        )

    file.write(
        "Selected PCs: "
        + str(
            (
                selected_indices
                + 1
            ).tolist()
        )
        + "\n"
    )

    file.write(
        f"Number selected: "
        f"{len(selected_indices)}\n"
    )

    file.write(
        f"Intercept: "
        f"{intercept:.6f}\n"
    )

    file.write(
        f"Weight norm: "
        f"{weight_norm:.6f}\n\n"
    )


    # ------------------------------------------------------------------
    # Part 6
    # ------------------------------------------------------------------

    file.write(
        "SYNTHETIC IMAGE RANGE\n"
    )

    file.write(
        "-" * 60
        + "\n"
    )

    file.write(
        "Requested range: "
        "0.5 to 5.5\n"
    )

    file.write(
        f"Prediction range on "
        f"original images: "
        f"{prediction_min:.4f} "
        f"to "
        f"{prediction_max:.4f}\n"
    )


    if len(outside_targets) == 0:

        file.write(
            "All requested targets were "
            "inside the prediction range.\n"
        )

    else:

        file.write(
            "Targets outside range: "
            + str(
                outside_targets.tolist()
            )
            + "\n"
        )

        file.write(
            "Replacement targets: "
            + str(
                np.round(
                    safe_target_ratings,
                    4,
                ).tolist()
            )
            + "\n"
        )


# ======================================================================
# FINAL TERMINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("ANALYSIS FINISHED")
print("=" * 70)


print(
    f"\nNumber of original images: "
    f"{n}"
)


print(
    f"PCs explaining at least "
    f"{VARIANCE_THRESHOLD:.0%} variance: "
    f"{n_components_selected}"
)


print(
    "PCs selected by forward selection:",
    selected_indices + 1,
)


print(
    "Number of selected PCs:",
    len(selected_indices),
)


print(
    f"Baseline CV MSE: "
    f"{baseline_cv_mse:.4f}"
)


if selection_history:

    print(
        f"Final CV MSE: "
        f"{selection_history[-1]:.4f}"
    )


print(
    f"Regression weight norm: "
    f"{weight_norm:.6f}"
)


print(
    f"Original-image prediction range: "
    f"{prediction_min:.4f} "
    f"to "
    f"{prediction_max:.4f}"
)


if len(outside_targets) == 0:

    print(
        "The requested synthetic range "
        "0.5-5.5 was inside the "
        "prediction range."
    )

else:

    print(
        "The requested synthetic range "
        "0.5-5.5 extended beyond "
        "the prediction range."
    )

    print(
        "A replacement set of "
        "synthetic faces was generated."
    )


print(
    "\nFiles saved:"
)

print(
    "- combined_ratings.csv"
)

print(
    "- model_predictions.csv"
)

print(
    "- analysis_summary.txt"
)

print(
    "- rating histograms"
)

print(
    "- PCA figures"
)

print(
    "- forward_selection_cv.png"
)

print(
    "- synthetic face images"
)