import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# Load books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load document embeddings
raw_documents = TextLoader("tagged_description.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# Recommendation logic
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs

# Formatting the output
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Dropdown options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Gradio UI
with gr.Blocks(theme=gr.themes.Base(), title="Book Recommender") as dashboard:
    gr.Markdown("""
        # 📚 <span style='color:#4F46E5'>Semantic Book Recommender</span>
        Discover amazing books based on your feelings and interests!
    """)

    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(label="🔍 Describe the kind of book you want:", placeholder="e.g., A mystery with deep emotions")
            category_dropdown = gr.Dropdown(choices=categories, label="📂 Select category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="🎭 Select mood:", value="All")
            submit_button = gr.Button("✨ Recommend Books")

            gr.Examples(
                examples=[
                    ["A journey of hope and redemption", "Fiction", "Happy"],
                    ["A thrilling adventure with plot twists", "Fiction", "Surprising"],
                    ["A touching story about grief and growth", "Nonfiction", "Sad"]
                ],
                inputs=[user_query, category_dropdown, tone_dropdown]
            )
        with gr.Column(scale=2):
            # Prefer local video if available; fallback to YouTube
            try:
                gr.Video("intro.mp4", label="🎥 How this works")
            except:
                gr.HTML("""
                <iframe width="100%" height="315"
                    src="https://www.youtube.com/embed/eg5I2UrpU8A"
                    title="Recommender Video"
                    frameborder="0" allowfullscreen></iframe>
                """)

    gr.Markdown("## 📖 Recommendations Just for You")
    output = gr.Gallery(label="Recommended Books", columns=4, rows=2, height="auto")

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Launch with PWA and optional favicon
if __name__ == "__main__":
    dashboard.launch(
        share=True,  # Generates a public URL
        server_name="0.0.0.0",  # Optional, for local network access
        server_port=7860,  # Optional, change port
        favicon_path="book-icon.png",  # Optional, custom icon
        pwa=True,  # Optional, enables Progressive Web App
        debug=True  # Optional, shows detailed error tracebacks
    )

