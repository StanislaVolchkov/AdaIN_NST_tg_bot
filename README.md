# Neural Style Transfer in telegram bot

   В данном репозитории представлено интегрированное в бота средство переноса стиля. Перенос стиля осуществляется посредством работы различных нейронных сетей. В зависимости от поставленных задач используется определенная архитектура.

## NST PART 
### AdaIN 

Архитектура, впервые описанная в 2017 в статье X. Huang and oth "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", представляет собой универсальную сверточную сеть способную в режиме реального времени переносить стиль с одного изображения на другое без ограничений в количестве доступных стилей. Сеть состоит из "енкодера" с первыми слоями VGG-19 вплоть до relu4-1 и симметричных ей слоях "декодера". Главной особенностью сети стал слой Adaptive Instance Normalization, который вычисляет параметры нормализации на основе поданных изображений, что и позволяет избавиться от ограниченности стилей. Также имеются функции контроля степени переноса стиля и смешивание нескольких.

В собственной реализации я решил изменить mse loss для стиля на l1 loss руководствуясь следующей логикой: 
    MSE loss будет сильнее штрафовать сеть за несоответсвие каких то отдельных пикселей, а само значение функции потерь будет сильно выше. При правильно подобранных для соответствующего датасета гиперпараметрах лоссов(в нашем случае 1 и 10 для контента и стиля соответственно), проблем c переносом стиля не возникает, в отличие от восстановления исходного изображения. В связи с этим было принято решение пожертвовать незаметным человеческому глазу несоответствием стилю для скорейшей сходимости лосса содержимого.
Также по аналогии с GAN добавить в архитектуру Гауссовский шум.  

Примеры работы модели:
![alt text](images/example.png?raw=true)

Некоторые заметки и выводы, сделанные в ходе работы:
* Bilinear interpolation не дает лучших результатов 
* Average Pooling смазывает изображение(что логично), а также сильно замедляет проход сети(что для меня стало неожиданным, но тоже логично)
* Гиперпараметры для различных датасетов действительно сложно подобрать обычному пользователю. Так что лучшим решением будет использовать оригинальные наборы.
* Обьективно сложно оценить влияние нововведений, но в любом случае я считаю, что удалось добиться неплохих результатов. А сама оригинальная архитектура доведена до идеала и на сегодняшний день сложно сильно ее улучшить, не изменяя основной концепции.

Note: В файле models/nst_model.py представлена обрезанная версия программы, созданная исключительно под инференс. В программе отсутвуют функции: обаботки датасета для обучения, самого обучения и его контроля. 

### MUNIT
Архитектура GAN разработанная и представленная компанией NVIDIA в 2017. Позволяет переносить неявные признаки одного изображения на другие. Присутствует слой AdaIN. В разработке

## BOT PART

Весь код представлен в файле telegram_bot.py. Бот построен на асинхронной системе с помощью фреймворка aiogram. Также приложение развернуто на Веб-хостинге Heroku с помощью webhook стратегии во избежание засыпания после инактива. Для запуска бота созданы два файла requirements.txt и Procfile с версиями библиотек необходимых для корректной работы и служебной информации о запуске для самого хостинга соответсвенно.

### requiremets.txt
    aiogram==2.19
    Pillow==7.1.2
    https://download.pytorch.org/whl/cpu/torch-1.10.0%2Bcpu-cp37-cp37m-linux_x86_64.whl
    https://download.pytorch.org/whl/cpu/torchvision-0.11.1%2Bcpu-cp37-cp37m-linux_x86_64.whl

В связи с ограничением предоставляемой памяти в 500МБ и бесмысленности установки CUDA версий на CPU-only сервис, версии для pytorch==1.10.0 и torchvision==1.10.1 представлены ссылкам на соответсвующие ресурсы для загрузки исключительно необходимых компонентов. Конечный размер сборки составляет 315 МБ.

### Procfile
    web: python3 telegram_bot.py

## Планируемые обновления 

* Более гибкая настройка стилизации AdaIN(Смешивание двух стилей при переносе на изображение) https://github.com/xunhuang1995/AdaIN-style

* MUNIT:
https://github.com/NVlabs/MUNIT github с MUNIT

https://medium.com/codex/review-munit-multimodal-unsupervised-image-to-image-translation-gan-10e2c08a1b6e  статья по MUNIT 

Различные углубления munit:

https://arxiv.org/pdf/2007.15651.pdf

https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_TransGaGa_Geometry-Aware_Unsupervised_Image-To-Image_Translation_CVPR_2019_paper.pdf

https://openaccess.thecvf.com/content_ECCV_2018/papers/Aaron_Gokaslan_Improving_Shape_Deformation_ECCV_2018_paper.pdf

https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang_Generative_Semantic_Manipulation_ECCV_2018_paper.pdf

+ Туториал по использованию предобученного CycleGan:
https://proglib.io/p/ispolzuem-cyclegan-dlya-primeneniya-stilya-k-video-poluchennomu-s-veb-kamery-2021-06-08

* Improved Optimization https://distill.pub/2018/differentiable-parameterizations/#section-styletransfer

* Additional Constraints https://distill.pub/2018/differentiable-parameterizations/#section-styletransfer

* Feature Visualization https://distill.pub/2017/feature-visualization/

* Post-processing https://github.com/titu1994/Neural-Style-Transfer

* https://github.com/ycjing/Neural-Style-Transfer-Papers

* SUPRES https://www.libhunt.com/r/pytorch-AdaIN

* DEEP PHOTO https://habr.com/ru/post/402665/ 

https://kushaj.medium.com/all-you-need-for-photorealistic-style-transfer-in-pytorch-acb099667fc8 статья 2019г.

## References

https://github.com/irasin/Pytorch_AdaIN

https://github.com/EugenHotaj/pytorch-generative


