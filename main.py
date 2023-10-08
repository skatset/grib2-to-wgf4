import logging
import os
import argparse
import re
import struct
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp
import asyncio
import aiofiles
import bz2
import multiprocessing

import numpy as np
import requests
import cfgrib
from collections import namedtuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

Grib2Data = namedtuple('Grib2Data', ['latitude_min', 'latitude_max', 'latitude_step',
                                     'longitude_min', 'longitude_max', 'longitude_step',
                                     'multiplier', 'data', 'predict_date'])


def extract_grib2_data(file_path: str) -> Grib2Data:
    # NOTE: хорошо бы хранить в памяти всё и передавать, диск лишняя сущность, в но в этом API библиотеки нет готового
    # способа работы с памятью. Можно переписать, если будет необходимость ускорить общее время работы
    # NOTE: мне нужна только часть датасета, возможно для ускорения доставать только его (это еще нужно проверить)
    # TODO: почему-то выдает ошибку "Ignoring index file '..._tot_prec.grib2.923a8.idx' older than GRIB file", нужно
    # либо добавить backend_kwargs={"indexpath": ""}, чтобы игнорировать эту ошибку, либо удалять файлы индексов перед
    # использованием (потому что они не используются, если уже созданы в предыдущем запуске). Нужно замерять как будет
    # быстрее и есть ли разница
    datasets = cfgrib.open_datasets(file_path)
    # TODO: нужно проверить поля в файле, что они соответствуют ожидаемому нам формату, для этого задания я это опущу
    ds = datasets[0]  # TODO проверить на количество и обработать как ошибку
    # TODO: нужно сделать проверку, что вычисленный шаг сетки соблюдается для всех значений
    multiplier = 1000000

    if len(ds.tp.to_numpy().shape) == 2:
        # NOTE: почему-то 07.10.2023 после 13 часов последняя 48ая запись получилась с 1 размерностью
        result_data = ds.tp.to_numpy()
    elif len(ds.tp.to_numpy().shape) == 3:
        # NOTE: мне приходят предсказания в 0, 15, 30 и 45ую минуту. В примере выгрузки в задании было показано, что за
        # 000 час тоже есть сохранение результатов. Это означает, что за каждый полученный час я должен брать последнее
        # предсказанное значение (за 45ую минуту). Это хорошо бы еще уточнить голосом/текстом
        result_data = ds.tp.to_numpy()[-1, :, :]
    else:
        # TODO: обработать ошибку
        return None
    pattern = r".*(\d{4})(\d{2})(\d{2})(\d{2})_(\d{3})_2d_tot_prec.grib2"
    match = re.match(pattern, file_path)
    if match:
        year, month, day, hour, offset = map(int, match.groups())
        date = datetime(year, month, day, hour)
        date += timedelta(hours=offset)
    else:
        # TODO: обработать ошибку
        return None
    return Grib2Data(latitude_min=int(ds.latitude.min() * multiplier),
                     latitude_max=int(ds.latitude.max() * multiplier),
                     latitude_step=int((ds.latitude.max() - ds.latitude.min()) * multiplier / len(ds.latitude)),
                     longitude_min=int(ds.longitude.min() * multiplier),
                     longitude_max=int(ds.longitude.max() * multiplier),
                     longitude_step=int((ds.longitude.max() - ds.longitude.min()) * multiplier / len(ds.longitude)),
                     multiplier=multiplier, data=result_data, predict_date=date)


def run_read_extracted_and_transform(extract_dir: str, converted: str, file_urls: List[str]) -> None:
    logger.info("Началась конвертация файлов")
    grib_data_list = []
    for url in file_urls:
        grib_data_list.append(extract_grib2_data(os.path.join(extract_dir, file_name_from_url(url))))
    for (i, d) in enumerate(grib_data_list):
        if i != 0:
            if d.latitude_step != grib_data_list[i - 1].latitude_step \
                    or d.longitude_step != grib_data_list[i - 1].longitude_step \
                    or d.latitude_max != grib_data_list[i - 1].latitude_max \
                    or d.latitude_min != grib_data_list[i - 1].latitude_min \
                    or d.longitude_max != grib_data_list[i - 1].longitude_max \
                    or d.longitude_min != grib_data_list[i - 1].longitude_min:
                # TODO: обработать ошибку
                continue
        output_dir = os.path.join(converted, d.predict_date.strftime("%Y%m%d_%H:%M_%s"))
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "PRATE.wgf4")
        with open(filepath, 'wb') as file:
            # NOTE: В ТЗ попросили сначала указать шаг по горизонтали, потом по вертикали. По горизонтали означает
            # справа налево (нагуглил в нескольких беседах и сам логически пришел, что идем вдоль горизонта), но
            # это вступает в противоречие с порядком latitude и longitude указанными ранее, ведь latitude это снизу
            # вверх. Этот момент нужно уточнить при личном общении, скорее всего здесь опечатка в ТЗ
            header = struct.pack('7i f', d.latitude_min, d.latitude_max, d.longitude_min, d.longitude_max,
                                 d.longitude_step, d.latitude_step, d.multiplier, -100500.0)
            data_to_write = prepare_grib2_data(d.data, grib_data_list[i - 1].data if i != 0 else None)
            file.write(header + data_to_write.flatten().tobytes())
    logger.info("Конвертация закончилась успешно")


def prepare_grib2_data(data: np.ndarray, previous_data: Optional[np.ndarray]) -> np.ndarray:
    if previous_data is not None:
        result_data = data - previous_data
        result_data[np.isnan(previous_data)] = data[np.isnan(previous_data)]
        return np.nan_to_num(result_data, nan=-100500.0)
    else:
        return np.nan_to_num(data, nan=-100500.0)


def file_name_from_url(url: str) -> str:
    filename = url.split("/")[-1]
    return filename[:-4]  # Убираем .bz2 из имени файла


async def fetch_file(session: aiohttp.ClientSession, url: str, extract_dir: str) -> None:
    async with session.get(url) as response:
        if response.status == 200:
            file_data = await response.read()
            filepath = os.path.join(extract_dir, file_name_from_url(url))
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(bz2.decompress(file_data))
            logger.info(f"Извлечен файл: {filepath}")


def download_and_extract(extract_dir: str, file_urls: List[str]) -> None:
    async def async_process() -> None:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_file(session, url, extract_dir) for url in file_urls]
            await asyncio.gather(*tasks)

    asyncio.run(async_process())


def split_by_chunk_and_run_download_and_extract_in_parallel(extract_dir: str, processes: int, file_urls: List[str]) -> None:
    # предотвращаем случай, когда есть остаток от деления и создается на 1 chunk больше
    chunk_size = len(file_urls) // processes + 1
    chunks = [file_urls[i:i + chunk_size] for i in range(0, len(file_urls), chunk_size)]

    processes = []
    for i in range(len(chunks)):
        process = multiprocessing.Process(target=download_and_extract, args=(extract_dir, chunks[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def fetch_file_list_and_run(args: argparse.Namespace) -> None:
    # NOTE: я использую requests вместо асинхронщины конкретно в этом месте, потому что мне нужно выполнить ровно 1
    # запрос и до его выполнения программа не может быть продолжена дальше, здесь чисто синхронный код
    # TODO: убрать warning NotOpenSSLWarning: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
    response = requests.get(args.url)
    if response.status_code != 200:
        logger.error(f"Не могу получить содержимое по запросу {args.url}, код ошибки {response.status_code}")
        return

    page_content = response.text
    file_urls = [
        args.url + line.split('"')[1]
        for line in page_content.split("\n")
        if line.startswith("<a href=") and "regular-lat-lon" in line
    ]

    # код для отладки, использую минимум значений
    # TODO: удалить перед показом
    # file_urls = file_urls[:2]

    split_by_chunk_and_run_download_and_extract_in_parallel(args.extract_dir, args.processes, file_urls)
    run_read_extracted_and_transform(args.extract_dir, args.converted_dir, file_urls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачивание файлов grib2, конвертация их в формат wgf4")
    parser.add_argument("--url",
                        default="https://opendata.dwd.de/weather/nwp/icon-d2/grib/12/tot_prec/",
                        help="URL сайта с файлами, по умолчанию https://opendata.dwd.de/weather/nwp/icon-d2/grib/12/tot_prec/")
    parser.add_argument("--extract_dir", default="extracted",
                        help="Каталог для извлеченных файлов, по умолчанию: extracted")
    parser.add_argument("--converted_dir", default="icon_d2",
                        help="Каталог для хранения сконвертированных файлов, по умолчанию: icon_d2")
    parser.add_argument("--processes", type=int, default=4,
                        help="Количество процессов для параллельной обработки, по умолчанию 4")
    args = parser.parse_args()

    os.makedirs(args.extract_dir, exist_ok=True)
    os.makedirs(args.converted_dir, exist_ok=True)

    fetch_file_list_and_run(args)


if __name__ == "__main__":
    main()
