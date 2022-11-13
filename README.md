## Сервис по идентификации гренландских китов на изображении от команды "Не GigaFlex"

### Обучение модели
Создаём виртуальную среду и устанавливаем зависимости из `requirements.txt`.
В конфиге прописываем пути к своим данным train_Arcface/config/base.yml.

Далее запускаем обучение:
```python
python main.py --cfg "config/base.yml"
```

### Запуск сервиса
Переносим обученную и сконвертированную модель в `PATH2PROJECT/service/models/whale_recognition/model.pt`. 

Устанавливаем необходимый зависимости:
```
pip install -r requirements.txt
```
Запускаем сервис:
```bash
streamlit run main.py
```