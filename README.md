# TODO
- запись содержимого в wgf4, по папкам
- обработка ошибок
- сбор логов в единый логгер, структурированные логи
- тесты
- Docker
- типизация
- комментарии к функциям

# Открытые вопросы
- будут ли блокировать за слишком частые скачивания с одного IP?
  - Во время локальных тестов я такое не встречал, если будет такое проявляться, то нужно будет пользоваться прокси,
писать задержку между запросами в пару секунд со случайным сдвигом в пару секунд
  - да и запуск этого скрипта подразумевается раз в день, о наличии новых можно проверять отдельно или подписаться на
RSS
- не понимаю что содержится в icosahedral
  - пока считываю только другой файл, regular-lat-lon, кажется в icosahedral отдельный формат, в нем в 1.43 
(примерно в корень из 2) раз больше данных, многие значения похожи, видится, что можно обойтись только одним форматом