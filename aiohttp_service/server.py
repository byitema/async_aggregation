import asyncio
import aiohttp
import configparser
import io
import numpy as np
import json
from aiohttp import web

routes = web.RouteTableDef()


async def send_to_processing(session, url, image, model):
    image_duplicate = io.BytesIO(image.read())
    image.seek(0)

    form_data = aiohttp.FormData()
    form_data.add_field(name='image', value=image_duplicate)
    form_data.add_field(name='model', value=model)
    async with session.post(url, data=form_data) as response:
        result = await response.text()
        return result


def geometry_mean(vectors):
    single_geometry_means = []
    for vector in vectors:
        vector = json.loads(vector)
        single_geometry_means.append(np.exp(np.mean(np.log(vector))))

    result = np.exp(np.mean(np.log(single_geometry_means)))

    return result


@routes.post('/process_image')
async def process_image(request: web.Request):
    data = await request.post()
    image = data['image']

    file = open('config.ini', 'r')
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read_file(file)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for key in config['urls']:
            tasks.append(asyncio.ensure_future(send_to_processing(session, config.get('urls', key), image.file, key)))

        results = await asyncio.gather(*tasks)

    return web.Response(text=str(geometry_mean(results)), content_type="text/html")


app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app)
