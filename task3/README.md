# Раскрашиватель карт глубины для датасета [Texel-BodyScan-Dataset](https://github.com/m-krivov/Texel-BodyScan-Dataset)

## Как использовать
1. Склонировать репозиторий
2. (Опц.) скомпилировать разделяемую библиотеку и шейдеры, используя скрипт ./vulkan/build.sh
3. Установить зависимости для Python, используя uv sync
4. Использовать python depth_pipeline.py --dataset dataset/Part1 --strength 6 --vulkan
   *   Настройка сила шумоподваления лучше всего от 4 до 6
   *   Если не получается подключить vulkan, то, возможно, стоит убрать эту опцию :(
6. Собрать видео, используя ./concat.sh frames out.mp4 (Требуется ffmpeg!)
