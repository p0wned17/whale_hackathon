import base64
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import PIL
import streamlit as st
import torch
from PIL import Image
from whale_recognition import model as whale_model
from whale_recognition import recognition, utils

IMAGES_DIR = Path(__file__).parent.parent.absolute() / "database/images"
IMAGES_DIR.mkdir(exist_ok=True, parents=True)
JSON_METADATA_PATH = Path(__file__).parent.parent.absolute() / "database/metadata.json"
EMBEDDINGS_PATH = (
    Path(__file__).parent.parent.absolute() / "database/all_embeddings.npy"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = torch.jit.load('src/model_2.pt')
model = whale_model.get_model()
model.to(device)


st.set_page_config(
    page_title="Добавление новых особей китов",
)

st.sidebar.markdown(
    "Данная страница нужна для добавления новой особи в базу для распознавания в последующих разах."
)

st.markdown(
    "<h2 style='text-align: center; color: black; font-size: 200%'>Добавление особи </h2>",
    unsafe_allow_html=True,
)


def upload_image_ui():
    uploaded_images = st.file_uploader(
        "Пожалуйста, выберите одно или несколько фото особи кита:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Нажмите, чтобы загрузить фото с китом",
    )
    return uploaded_images


def process_images(list_images: list, title: str):
    my_bar = st.progress(0)
    metadata = utils.get_metadata(JSON_METADATA_PATH)
    col1, col2, col3 = st.columns(3)
    if len(list_images) > 0:
        add_button = col2.button("Добавить", disabled=False)
    else:
        add_button = col2.button("Добавить", disabled=True)
    if add_button:
        with col1:
            st.write(" ")
        with col2:
            proccess_image = st.empty()
        with col3:
            st.write(" ")

        count_images = len(list_images)

        if count_images > 0:
            with col2:
                with open("assets/whale.gif", "rb") as file:
                    gif = file.read()
                    data_url = base64.b64encode(gif).decode("utf-8")
                    proccess_image = st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}">',
                        unsafe_allow_html=True,
                    )

        with col2:
            st.empty()
        with st.expander("Разверните для просмотра деталей"):
            embeddings_db = utils.get_numpy_db(EMBEDDINGS_PATH)

            if metadata.get(f"{title}_index") is not None:
                embedding_idx = metadata[f"{title}_index"]
                old_embedding = embeddings_db[embedding_idx]
                exist = True
            else:
                exist = False
                metadata[f"{title}_images"] = []
                old_embedding = None
                metadata[f"{title}_index"] = len(embeddings_db) - 1

            numpy_list_images = [utils.bytes_to_numpy(image) for image in list_images]

            embedding = recognition.compute_mean_embedding(
                model=model,
                images=numpy_list_images,
                device=device,
                old_embedding=old_embedding,
            )
            if exist == False:

                embeddings_db = np.r_[embeddings_db, [embedding]]
            else:
                embedding_idx = metadata[f"{title}_index"]
                embeddings_db[embedding_idx] = embedding

            print(embeddings_db.shape)
            # embeddings_db = np.vstack((embeddings_db, embedding))
            np.save(str(EMBEDDINGS_PATH), embeddings_db)

            for image in list_images:

                image_path = IMAGES_DIR / f"{title}/{uuid4()}.jpg"
                image_path.parent.mkdir(exist_ok=True, parents=True)
                metadata[f"{title}_images"].append(image_path.__str__())
                image_path.write_bytes(image.getbuffer())

            with JSON_METADATA_PATH.open(mode="w") as f:
                json.dump(metadata, f)
                # with open('data.json', 'w') as f:

            proccess_image.success("Изображения успешно добавлены в базу!")


def main():
    st.write("Здесь вы можете загрузить фото новой особи гренладского кита")
    title = st.text_input("Введите ID кита:", "")

    list_images = upload_image_ui()
    process_images(list_images, title)


if __name__ == "__main__":
    main()
