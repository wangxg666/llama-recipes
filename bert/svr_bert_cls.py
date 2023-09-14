import fire
import numpy as np

from utils_ import *

import numpy as np


def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x



def main(model_name: str = '',
         fine_tuning_model: str = '',
         port: int = 1302):

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = torch.load(fine_tuning_model)
    model.eval()

    async def predict_relevance(request):
        obj = await request.json()
        query = obj['query']
        knn_queries = obj['knn_queries']

        texts = [
            [query, knn_query] for knn_query in knn_queries
        ]

        input_ids = encode_fn(texts, tokenizer)
        out = model(input_ids.to(device), token_type_ids=None, attention_mask=(input_ids > 0).to(device))
        logits = out.logits
        logits = logits.detach().cpu().numpy()
        pred_labels = np.argmax(logits, axis=1)
        pred_labels = [index2label[i] for i in pred_labels.tolist()]
        pred_scores = [x[0] for x in softmax(logits).tolist()]

        return web.json_response(data={'pred': pred_labels, 'scores': pred_scores})


    from aiohttp import web

    app = web.Application()
    app.add_routes([web.post('/do', predict_relevance)])
    web.run_app(app, port=port, access_log=None)


if __name__ == '__main__':
    fire.Fire(main)