import os
import argparse
import aiohttp
import asyncio
import aiofiles
import bz2
import multiprocessing
import requests
import cfgrib
from collections import namedtuple


Grib2Data = namedtuple('Grib2Data', ['latitude_min', 'latitude_max', 'latitude_step',
                                     'longitude_min', 'longitude_max', 'longitude_step',
                                     'multiplier', 'data'])

def extract_grib2_data(file_path):
    # NOTE: хорошо бы хранить в памяти всё и передавать, диск лишняя сущность, в но в этом API библиотеки нет готового
    # способа работы с памятью. Можно переписать, если будет необходимость ускорить общее время работы
    # NOTE: мне нужна только часть датасета, возможно для ускорения доставать только его (это еще нужно проверить)
    # TODO: почему-то выдает ошибку "Ignoring index file '..._tot_prec.grib2.923a8.idx' older than GRIB file", нужно разобраться
    datasets = cfgrib.open_datasets(file_path)
    # TODO: нужно проверить поля в файле, что они соответствуют ожидаемому нам формату, для этого задания я это опущу
    ds = datasets[0]  # TODO проверить на количество и обработать как ошибку
    # TODO: нужно сделать проверку, что вычисленный шаг сетки соблюдается для всех значений
    multiplier = 1000000

    if len(ds.tp.to_numpy().shape) == 2:
        # NOTE: почему-то 07.10.2023 после 13 часов последняя 48ая запись получилась с 1 размерностью
        print(ds)
        result_data = ds.tp.to_numpy()
    elif len(ds.tp.to_numpy().shape) == 3:
        # NOTE: мне приходят предсказания в 0, 15, 30 и 45ую минуту. В примере выгрузки в задании было показано, что за
        # 000 час тоже есть сохранение результатов. Это означает, что за каждый полученный час я должен брать последнее
        # предсказанное значение (за 45ую минуту). Это хорошо бы еще уточнить голосом/текстом
        result_data = ds.tp.to_numpy()[-1, :, :]
    else:
        # TODO: обработать ошибку
        return
    return Grib2Data(latitude_min=int(ds.latitude.min() * multiplier), latitude_max=int(ds.latitude.max() * multiplier),
                     latitude_step=int((ds.latitude.max() - ds.latitude.min()) * multiplier / len(ds.latitude)),
                     longitude_min=int(ds.longitude.min() * multiplier), longitude_max=int(ds.longitude.max() * multiplier),
                     longitude_step=int((ds.longitude.max() - ds.longitude.min()) * multiplier / len(ds.longitude)),
                     multiplier=multiplier, data=result_data)


def run_read_extracted_and_transform(extract_dir, file_urls):
    # TODO: читать файлы параллельно
    grib_data_list = []
    for url in file_urls:
        grib_data_list.append(extract_grib2_data(os.path.join(extract_dir, file_name_from_url(url))))
    # TODO: нужно проверить, что у двух соседних совпадает минимальные и максимальные координаты и шаг
    # TODO: вычитать из следующего часа предыдущий
    # TODO: реализовать запись в формате wgf4
    print(grib_data_list)


def file_name_from_url(url):
    filename = url.split("/")[-1]
    return filename[:-4]  # Убираем .bz2 из имени файла


async def fetch_file(session, url, extract_dir):
    async with session.get(url) as response:
        if response.status == 200:
            file_data = await response.read()
            filepath = os.path.join(extract_dir, file_name_from_url(url))
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(bz2.decompress(file_data))
            print(f"Извлечен файл: {filepath}")


def run_download_and_extract(extract_dir, file_urls):
    async def download_and_extract():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_file(session, url, extract_dir) for url in file_urls]
            await asyncio.gather(*tasks)

    asyncio.run(download_and_extract())


def fetch_file_list_run_processing_by_chunk(args):
    # NOTE: я использую requests вместо асинхронщины конкретно в этом месте, потому что мне нужно выполнить ровно 1
    # запрос и до его выполнения программа не может быть продолжена дальше, здесь чисто синхронный код
    # TODO: убрать warning NotOpenSSLWarning: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
    response = requests.get(args.url)
    if response.status_code != 200:
        print(f"Не могу получить содержимое по запросу {args.url}, код ошибки {response.status_code}")
        return

    page_content = response.text
    file_urls = [
        args.url + line.split('"')[1]
        for line in page_content.split("\n")
        if line.startswith("<a href=") and 'regular-lat-lon' in line
    ]

    # код для отладки, использую минимум значений
    # TODO: удалить перед показом
    # file_urls = file_urls[:2]

    # предотвращаем случай, когда есть остаток от деления и создается на 1 chunk больше
    chunk_size = len(file_urls) // args.processes + 1
    chunks = [file_urls[i:i + chunk_size] for i in range(0, len(file_urls), chunk_size)]

    processes = []
    for i in range(len(chunks)):
        process = multiprocessing.Process(target=run_download_and_extract, args=(args.extract_dir, chunks[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    run_read_extracted_and_transform(args.extract_dir, file_urls)


def main():
    parser = argparse.ArgumentParser(description="Скачивание файлов grib2, конвертация их в формат wgf4")
    parser.add_argument("--url",
                        default="https://opendata.dwd.de/weather/nwp/icon-d2/grib/12/tot_prec/",
                        help="URL сайта с файлами, по умолчанию https://opendata.dwd.de/weather/nwp/icon-d2/grib/12/tot_prec/")
    parser.add_argument("--extract_dir", default="extracted",
                        help="Каталог для извлеченных файлов, по умолчанию: extracted")
    parser.add_argument("--processes", type=int, default=4,
                        help="Количество процессов для параллельной обработки, по умолчанию 4")
    args = parser.parse_args()

    os.makedirs(args.extract_dir, exist_ok=True)

    fetch_file_list_run_processing_by_chunk(args)


if __name__ == "__main__":
    main()
