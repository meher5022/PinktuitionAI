import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stApp {
    background-color: #FFFDFD;
}

h1,h2,h3 {
    font-family: Arial;
}

.stButton > button {
    font-size: 22px !important;
    font-weight: 900 !important;
    color: white !important;

    background: linear-gradient(to right,#F8AFC8,#F39ABB);
    border-radius: 10px;
    height: 2em;
    border: none;
}
            


.stButton>button:hover{
    background:#EE8EB1;
    box-shadow:0px 6px 16px rgba(232,84,128,0.35);
}

.stButton>button:hover{
    background:#EE8EB1;
}

.stButton>button:hover{
    background:#C94F7C;
}

div[data-testid="stMetric"]{
    background-color:#FFF4F8;
    border-radius:15px;
    padding:15px;
    border:1px solid #F4D5E1;
}

section[data-testid="stSidebar"]{
    background-color:#FFF4F8;
}

</style>
""",
unsafe_allow_html=True)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pinktuition AI",
    page_icon="🧴",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_products():
    df = pd.read_csv("sephora_final.csv")

    df = df[
        (df["ingredients"].notna()) &
        (df["rating"].notna()) &
        (df["price_usd"].notna())
    ].copy()

    df["ingredients"] = df["ingredients"].astype(str).str.lower()

    return df


@st.cache_data
def load_reviews():
    reviews = pd.read_csv("sephora_reviews.csv")

    reviews = reviews.dropna(subset=["review_text"])

    reviews["review_text"] = reviews["review_text"].astype(str)
    reviews["review_title"] = (
        reviews["review_title"]
        .fillna("")
        .astype(str)
    )

    return reviews


df = load_products()
reviews = load_reviews()

# ============================================================
# PRICE TIERS
# ============================================================

def assign_price_tier(price):

    if pd.isna(price):
        return "unknown"

    elif price < 30:
        return "cheap"

    elif price < 80:
        return "mid_range"

    else:
        return "expensive"


df["price_tier"] = df["price_usd"].apply(assign_price_tier)

# ============================================================
# CONCERN PROFILES
# ============================================================

CONCERN_PROFILES = {
    "brightening":
        "vitamin c kojic acid glycolic acid niacinamide",

    "anti_aging":
        "retinol peptides vitamin c adenosine",

    "hydrating":
        "hyaluronic acid ceramide glycerin squalane",

    "acne_care":
        "salicylic acid niacinamide tea tree zinc",

    "sensitive_skin":
        "ceramide niacinamide aloe vera panthenol",
}

# ============================================================
# TF-IDF MODEL
# ============================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

tfidf.fit(df["ingredients"])

# ============================================================
# FUNCTIONS
# ============================================================

def recommend(
    concern,
    brand,
    price_tier,
    min_rating,
    top_n
):

    pool = df.copy()

    pool = pool[
        pool["rating"] >= min_rating
    ]

    if brand != "All":
        pool = pool[
            pool["brand_name"] == brand
        ]

    if price_tier != "all":
        pool = pool[
            pool["price_tier"] == price_tier
        ]

    pool = pool.reset_index(drop=True)

    if len(pool) == 0:
        return pd.DataFrame()

    query_vector = tfidf.transform(
        [CONCERN_PROFILES[concern]]
    )

    ingredient_vectors = tfidf.transform(
        pool["ingredients"]
    )

    pool["score"] = cosine_similarity(
        query_vector,
        ingredient_vectors
    ).flatten()

    # Scale scores relative to the best match

    if pool["score"].max() > 0:
        pool["score"] = (
            pool["score"] / pool["score"].max()
        )

    return (
        pool
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_n)
    )


def get_reviews(
    product_id,
    n_reviews=3
):

    product_reviews = reviews[
        reviews["product_id"] == product_id
    ]

    if len(product_reviews) == 0:
        return None

    return product_reviews.head(n_reviews)


# ============================================================
# TITLE
# ============================================================

st.markdown("""
<h1 style='text-align:center;'>
🧴🫧Pinktuition AI🫧🧴
</h1>
""", unsafe_allow_html=True)

st.markdown(
"<h2 style='text-align:center;'>"
"AI-powered skincare recommendations based on ingredient intelligence."
"</h2>",
unsafe_allow_html=True
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# 🌸 Your Skin Profile

Customize your skincare journey.
""")

concern = st.sidebar.selectbox(
    "Skin Concern",
    list(CONCERN_PROFILES.keys())
)

brand = st.sidebar.selectbox(
    "Brand",
    ["All"] +
    sorted(
        df["brand_name"]
        .dropna()
        .unique()
    )
)

price_tier = st.sidebar.selectbox(
    "Price Tier",
    [
        "all",
        "cheap",
        "mid_range",
        "expensive"
    ]
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    5.0,
    3.5
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    3,
    15,
    5
)

st.sidebar.divider()



st.sidebar.markdown(
    """
    ### Set your filters and click
    Recommend Me Products ✨ 
    """
)



# ============================================================
# MAIN BUTTON
# ============================================================

st.markdown(
"""
<h3 style='text-align:center;'>
Listen to your Pinktution 🧼
</h3>
""",
unsafe_allow_html=True
)

left, center, right = st.columns([1,2,1])

with center:
    run = st.button(
        "Recommend Me Products ✨ ",
        use_container_width=True
    )   

# ============================================================
# RESULTS
# ============================================================

if run:

    results = recommend(
        concern,
        brand,
        price_tier,
        min_rating,
        top_n
    )

    col1, col2 = st.columns([2, 1])

    # ========================================================
    # LEFT PANEL
    # ========================================================

    with col1:

        st.subheader("💆🏻‍♀️ Your Personalized Recommendations 💆🏻‍♀️")

        if results.empty:

            st.warning(
                "No matches found. "
                "Try adjusting the filters."
            )

        else:

            for _, row in results.iterrows():

                with st.container(border=True):

                    st.subheader(
                        row["product_name"]
                    )

                    st.write(
                        f"**Brand:** "
                        f"{row['brand_name']}"
                    )

                    st.write(
                        f"**Price Tier:** "
                        f"{row['price_tier']}"
                    )

                    st.write(
                        f"**Rating:** ⭐ "
                        f"{row['rating']}"
                    )

                    st.write(
                        f"**Price:** "
                        f"${row['price_usd']}"
                    )

                    score = row["score"] * 100

                    if score >= 70:
                        st.success(
                            f"Match Score: "
                            f"{score:.1f}%"
                        )
                    elif score >= 40:
                        st.warning(
                            f"Match Score: "
                            f"{score:.1f}%"
                        )
                    else:
                        st.error(
                            f"Match Score: "
                            f"{score:.1f}%"
                        )

                    # Matching Ingredients

                    query_words = set(
                        CONCERN_PROFILES[
                            concern
                        ].split()
                    )

                    product_words = set(
                        row["ingredients"].split()
                    )

                    common = list(
                        query_words.intersection(
                            product_words
                        )
                    )

                    if len(common) > 0:
                        st.write(
                            "**Matching Ingredients:**"
                        )
                        st.info(
                            ", ".join(common)
                        )

                    # Reviews

                    product_reviews = get_reviews(
                        row["product_id"]
                    )

                    if product_reviews is not None:

                        with st.expander(
                            "💬 Customer Reviews"
                        ):

                            for _, review in (
                                product_reviews
                                .iterrows()
                            ):

                                if (
                                    review[
                                        "review_title"
                                    ].strip()
                                    != ""
                                ):
                                    st.markdown(
                                        f"**{review['review_title']}**"
                                    )

                                review_rating = review["rating"]

                                if review_rating >= 4:
                                    st.success(review[
                                        "review_text"
                                    ])

                                elif review_rating >= 3:
                                    st.warning(review[
                                        "review_text"
                                    ])

                                else:
                                    st.error(review[
                                        "review_text"
                                    ])

                    else:
                        st.caption(
                            "No customer reviews available."
                        )

    # ========================================================
    # RIGHT PANEL
    # ========================================================

    with col2:

        st.subheader("Insights")

        if not results.empty:

            st.metric(
                "Best Match",
                f"{results['score'].max()*100:.1f}%"
            )

            st.metric(
                "Average Rating",
                round(
                    results["rating"]
                    .mean(),
                    2
                )
            )

            st.metric(
                "Average Price",
                f"${results['price_usd'].mean():.2f}"
            )

            st.metric(
                "Products Found",
                len(results)
            )

            st.divider()

            st.bar_chart(
                results
                .set_index(
                    "product_name"
                )["score"]
                * 100
            )

            csv = results.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="pink_results.csv",
                mime="text/csv"
            )

else:

    st.divider()

    st.markdown("""
    <div style="
    text-align:center;
    color:gray;
    padding:10px;
    background-color:#FFF4F8;
    border-radius:15px;
    ">

    <h3>
    Pinktuition AI 🫧
    </h3>

    <p>
    Built using TF-IDF, Cosine Similarity and NLP
    </p>

    </div>
    """, unsafe_allow_html=True)

