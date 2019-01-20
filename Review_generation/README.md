Все результаты и описания модели в юпитер ноутбуке:
lm_Shevtsov.ipynb (32.89 kB)

Код ноутбука  в python модуле.

lm_dataset.py (1.03 kB)
lm_model.py (5.08 kB)
lm_starter.py (3.94 kB)

Модель обучается на школьных сочинениях по произведению 'Евгений Онегин', модуль сам скачивает и парсит данные с сайта на компьютер и формирует дотасет, скачка занимает 2-3 минуты.

Если данные уже скачаны то после проставления ключа --download_dataset 0 программа будет собирать датасет из уже скаченной папки. Ключ --epoch получает на вход целое значение, число эпох, а --starting_word строку, начальное слово (если не будет этого ключа, то программа просто возьмет рандомное слово из словаря)


Оптимально для первого старта:

python lm_starter.py --epoch 5 -- starting_word 'роман'
(каждая эпоха в среднем 13-15 минут на среднемощном ноутбуке)


Пример самого простого запуска ( почти без обучения и с всего 10 итерациями по батчам, обычно их 340):

noname:proj anton.shevtsov$ python lm_starter.py --download_dataset 0 --epoch 1
Reading...
100%|███████████████████████████████| 306/306 [00:00<00:00, 11944.25it/s]
Done!
Words in texts: 249939
Unique words in texts: 25158
2018-11-22 09:48:04.248370: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Dataset generated
Training...
0it [00:00, ?it/s]Epoch 1 Batch 0 Loss 10.1329
10it [00:25, 2.56s/it]
Epoch 1 Loss 7.7305
Time taken for 1 epoch 25.882055044174194 sec

Done!
родители , не , так не не с . пушкин , в . . , , он жизни . . в . , . . в
