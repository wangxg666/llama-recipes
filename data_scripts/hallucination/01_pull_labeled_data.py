from pymongo import MongoClient


if __name__ == '__main__':
    mongo = MongoClient("mongodb://video.mongo.nb.com/?replicaSet=rs_video&readPreference=secondaryPreferred")
    col_label_data = mongo['aigc']['hallucination']

    for obj in col_label_data.find({'moderation.label': {'$in': [
            'Pair / Omission: key information',
            'Pair / Hallucinate with the original text (New Content is Generated)',
            'Pair / Deviations: new information created that wasn’t present in the original input'
        ]}}, {
            'moderation.label': 1,
            'doc_id': 1
        }):
        print(obj)