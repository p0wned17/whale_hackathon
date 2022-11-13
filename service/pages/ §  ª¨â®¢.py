import heapq
from glob import glob

import streamlit as st

PATH = "database_image/"

st.set_page_config(
    page_title="База китов",
)


def percent(len_images, percent=30):
    percent = len_images * 0.33
    return int(percent)


def normal_path(art):
    heapq.nsmallest(len(art), art)

    directory_names = []

    for name in art:
        directory_names.append(int(name.split("/")[-1]))

    return directory_names


def get_images(option):

    list_images = glob(f"{PATH}{option}/images/*", recursive=True)

    return list_images


st.markdown(
    "<h2 style='text-align: center; color: black; font-size: 300%'>🐳  База китов 🐳 </h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("Данная страница нужна для отображения галереи добавленных китов.")
col1, col2, col3 = st.columns(3)

art = glob(f"{PATH}*", recursive=True)

directory_names = normal_path(art)

with st.sidebar:
    option = st.selectbox(
        "Выберите класс гренладского кита",
        heapq.nsmallest(len(directory_names), directory_names),
    )

images = get_images(option)

len_images = len(images)

percent = percent(len_images)


with col1:
    st.image(images[0:percent])

with col2:
    st.image(images[percent : percent + percent])

with col3:
    st.image(images[percent + percent : percent + percent + percent])
