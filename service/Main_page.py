import base64
import io
import json
import os

import PIL
import streamlit as st
import torch
from PIL import Image
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.jit.load("src/model_copy_xz.pt")

model.to(device)

embed_list = np.load("src/model_copy_xz.npy")

with open("src/model_copy_xz.json") as f:
    file_content = f.read()
    idx_list = json.loads(file_content)


st.set_page_config(
    page_title="Главная страница",
)

st.markdown(
    "<h2 style='text-align: center; color: #34495e; font-size: 300%'>Главная страница</h2>",
    unsafe_allow_html=True,
)


def upload_image_ui():
    uploaded_images = st.file_uploader(
        "Пожалуйста, выберите файлы с изображениями:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Нажмите, чтобы загрузить фото с китом",
    )
    return uploaded_images


def process_images(list_images: list):
    my_bar = st.progress(0)

    col1, col2, col3 = st.columns(3)
    if col2.button("Загрузить изображение"):
        with col1:
            st.write(" ")
        with col2:
            proccess_image = st.empty()
        with col3:
            st.write(" ")

        count_images = len(list_images)

        all_walruses_count = 0
        coords_report_data = []
        all_walruses_data = []

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
            for i, image in enumerate(list_images):
                filename, extension = os.path.splitext(image.name)

                opencv_image = bytes_to_numpy(image)
                # print(opencv_image)
                emb = extractor_from_input_image(opencv_image, model, device)
                calculated_embed = calculate_embed(idx_list, embed_list, emb, 0.2)
                print(calculated_embed)

                if isinstance(opencv_image, np.ndarray):
                    with st.spinner(f"Проверяю китов на {i+1} изображении ..."):
                        type_of_whale = -1
                        if type_of_whale < 0:
                            st.write(
                                "Совпадение не было, предлагаем добавить новую особь!"
                            )
                            st.markdown(
                                '<center><a href="/Add_new_Whale" style="text-align: center;" target="_self">Добавить новую особь</a></center>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.image(
                                opencv_image,
                                channels="BGR",
                                caption=f"На изображении найден вид особи под номером: {type_of_whale}",
                            )
                else:
                    st.error(
                        "С изображением что-то не так, проверьте и загрузите заново!"
                    )
                my_bar.progress((i + 1) / count_images)
            with col1:
                st.write(" ")
            with col2:
                st.empty()
            with col3:
                st.write(" ")

            if count_images > 0:
                with col1:
                    st.write(" ")
                with col2:
                    proccess_image.empty()

                with col3:
                    st.write(" ")


def main():
    st.write("Здесь вы можете загрузить фото кита")
    list_images = upload_image_ui()
    process_images(list_images)


if __name__ == "__main__":
    main()
