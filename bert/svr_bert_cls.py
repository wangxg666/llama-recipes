import numpy as np

from utils_ import *


model = torch.load('')
model.eval()


async def predict_relevance(request):
    obj = await request.json()
    query = obj['query']
    knn_queries = obj['knn_queries']

    texts = [
        f'{query} [SEP] {knn_query}' for knn_query in knn_queries
    ]

    input_ids = encode_fn(texts)
    out = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids > 0).to(device))
    logits = out.logits
    logits = logits.detach().cpu().numpy()
    pred_labels = np.argmax(logits, axis=1)
    pred_labels = [index2label[i] for i in pred_labels.tolist()]

    return web.json_response(data={'pred': pred_labels})


from aiohttp import web

app = web.Application()
app.add_routes([web.post('/do', predict_relevance)])
web.run_app(app, port=1301, access_log=None)