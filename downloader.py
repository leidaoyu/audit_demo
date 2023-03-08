import requests
import os

_WINDOWS = os.name == 'nt'
COPY_BUFSIZE = 1024 * 1024 if _WINDOWS else 64 * 1024

header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                        'like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.71'}


def copyfileobj(fsrc, fdst, size, length=0):
    """
    copy data from file-like object fsrc to file-like object fdst
    :param fsrc: 被拷贝文件
    :param fdst: 目标文件
    :param size: 文件大小(B)
    :param length: 分块大小
    :return:
    """
    # Localize variable access to minimize overhead.
    if not length:
        length = COPY_BUFSIZE
    fsrc_read = fsrc.read
    fdst_write = fdst.write
    i = 0
    print('0.00% |', end='')
    while True:
        buf = fsrc_read(length)
        if not buf:
            break
        fdst_write(buf)
        i += length
        sp = i / size
        if sp > 1:
            sp = 1
        print(f'\r', format(sp * 100, '.2f'), '% |', '▊' * int(40 * sp), sep='', end='')
    print('\n')


def download_file(url, path, redown=0):
    """文件下载
    :param url: 下载文件url
    :param path: 保存路径 需要文件名和扩展名
    :param redown: 是否询问重新下载(0:询问 1:默认不重新下载 2:默认重新下载)
    :return: 成功:0 失败:1
    """
    print(url.split('/')[-1], '=>', path.split('/')[-1])
    if os.path.exists(path):
        if redown == 0:
            if input('文件已存在,是否重新下载? (y/n)') != 'y':
                return 0
        elif redown == 1:
            return 0
        os.remove(path)

    with requests.get(url, stream=True, headers=header) as r:
        with open(path + '.tmp', 'wb') as f:
            copyfileobj(r.raw, f, int(r.headers['Content-Length']))
    try:
        os.rename(path + '.tmp', path)
    except FileExistsError:
        return 1
    return 0


if __name__ == '__main__':
    print(download_file('https://mirrors.tuna.tsinghua.edu.cn/debian-cd/11.4.0/amd64/iso-cd/'
                        'debian-11.4.0-amd64-netinst.iso', 'test.iso'))

