import os
import argparse
import aiohttp
import asyncio
import aiofiles
import bz2
import multiprocessing
import requests


async def fetch_file(session, url, extract_dir):
    async with session.get(url) as response:
        if response.status == 200:
            file_data = await response.read()
            filename = url.split("/")[-1]
            filepath = os.path.join(extract_dir, filename[:-4])  # Убираем .bz2 из имени файла
            async with aiofiles.open(filepath, "wb") as file:
                await file.write(bz2.decompress(file_data))
            print(f"Извлечен файл: {filepath}")


async def download_and_extract(extract_dir, file_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_file(session, url, extract_dir) for url in file_urls]
        await asyncio.gather(*tasks)


def run_chunk_processing(extract_dir, file_urls):
    asyncio.run(download_and_extract(extract_dir, file_urls))


def fetch_file_list_run_processing_by_chunk(args):
    response = requests.get(args.url)
    if response.status_code != 200:
        print(f"Не могу получить содержимое по запросу {args.url}, код ошибки {response.status_code}")
        return

    page_content = response.text
    file_urls = [
        args.url + line.split('"')[1]
        for line in page_content.split("\n")
        if line.startswith("<a href=") and '<a href="../">' not in line
    ]

    # предотвращаем случай, когда есть остаток от деления и создается на 1 chunk больше
    chunk_size = len(file_urls) // args.processes + 1
    chunks = [file_urls[i:i + chunk_size] for i in range(0, len(file_urls), chunk_size)]

    processes = []
    for i in range(args.processes):
        process = multiprocessing.Process(target=run_chunk_processing,
                                          args=(args.extract_dir, chunks[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


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
